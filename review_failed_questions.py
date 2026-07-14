"""
review_failed_questions.py

Reviews only questions that failed retrieval and checks whether the question
itself is too vague to be a reliable evaluation item.

Input:
    results/failed_questions.jsonl

Recommended source for that file:
    results/evaluation_predictions.jsonl

For every failed question, the judge receives:
    - the question
    - the reference answer
    - the original gold span(s)
    - the retrieved contexts
    - the original gold chunk IDs
    - the retrieved chunk IDs

The script performs two layers of vagueness checking:

1. Fast rule-based screening
   Flags common vague constructions such as:
       "According to the report..."
       "What were the two objectives?"
       "What information is presented in Appendix V?"
       "What did the decision recognize?"
   when the question does not identify the report, program, agency, case,
   system, event, or other subject clearly enough.

2. LLM judge
   Determines whether a reader could identify the intended subject without
   already seeing the gold passage.

Possible classifications:
    TRUE_RETRIEVAL_FAILURE
    INCOMPLETE_GOLD_LABEL
    AMBIGUOUS_QUESTION
    BAD_REFERENCE_OR_GOLD_SPAN
    OTHER

A record can also receive:
    vague_question = true
    vagueness_reason = "..."
    suggested_rewrite = "..."

Suggested rewrites are diagnostics only. Human review is still required.

Usage:
    python review_failed_questions.py
    python review_failed_questions.py --path results/failed_questions.jsonl
    python review_failed_questions.py --out results/failed_questions_review.jsonl
"""

import argparse
import json
import re
import time
from pathlib import Path

from corpus import load_raw_docs
from eval_common import (
    get_gemini_client,
    call_gemini_json,
    span_text,
)

DEFAULT_PATH = Path("results/failed_questions.jsonl")
DEFAULT_DATASET_PATH = Path(
    "data/dataset/gov_report_sample_2k.json"
)
DEFAULT_REPORT_PATH = Path(
    "results/failed_questions_review.jsonl"
)
DEFAULT_SLEEP_SECONDS = 15


# ============================================================
# FAST VAGUENESS SCREEN
# ============================================================

GENERIC_NOUNS = {
    "report",
    "review",
    "decision",
    "study",
    "analysis",
    "document",
    "program",
    "system",
    "project",
    "proposal",
    "bill",
    "act",
    "meeting",
    "agreement",
    "appendix",
    "table",
    "figure",
    "draft",
    "effort",
    "initiative",
    "process",
    "requirements",
    "objectives",
    "issues",
    "information",
    "modifications",
    "changes",
    "findings",
    "recommendations",
    "comments",
    "individuals",
    "officials",
    "agency",
    "contractor",
}

VAGUE_START_PATTERNS = [
    r"^what (?:was|were|is|are|did|does) the ",
    r"^according to the (?:report|review|decision|study|analysis|chunk|passage)",
    r"^what information is presented in ",
    r"^what did the (?:report|review|decision|draft|study) ",
    r"^what were the (?:two|three|primary|main) ",
    r"^what kind of ",
    r"^what type of ",
]

DEICTIC_PATTERNS = [
    r"\bthe report\b",
    r"\bthe review\b",
    r"\bthe decision\b",
    r"\bthe study\b",
    r"\bthe draft report\b",
    r"\bthe contractor\b",
    r"\bthe individuals interviewed\b",
    r"\bthe meeting\b",
    r"\bthe program\b",
    r"\bthe system\b",
    r"\bthe bill\b",
    r"\bthe agreement\b",
    r"\bappendix [ivxlcdm]+\b",
    r"\bfigure \d+\b",
    r"\btable \d+\b",
]


def tokenize(text):
    return re.findall(
        r"[A-Za-z0-9][A-Za-z0-9'’-]*",
        text.lower(),
    )


def has_specific_anchor(question):
    """
    Looks for signs that the question names a distinct subject.

    Examples of anchors:
      - acronyms: EPA, VA, CMS, RS.AN-1
      - years: 2008, FY2014
      - named laws/programs: USA PATRIOT Act
      - proper-name sequences: Retirement Information Office
      - quoted phrases
    """
    if re.search(r"\b[A-Z]{2,}(?:[-.][A-Z0-9]+)*\b", question):
        return True

    if re.search(r"\b(?:19|20)\d{2}\b", question):
        return True

    if re.search(r"\bFY\s?\d{2,4}\b", question, re.I):
        return True

    if re.search(r"['“\"][^'”\"]{3,}['”\"]", question):
        return True

    proper_name_sequences = re.findall(
        r"\b(?:[A-Z][a-z]+(?:\s+|$)){2,}",
        question,
    )

    return bool(proper_name_sequences)


def fast_vagueness_check(question):
    """
    Conservative heuristic.

    It does not mark every short question as vague. It flags questions that
    combine generic references with no clear subject anchor.
    """
    normalized = " ".join(question.strip().split())
    lowered = normalized.lower()
    tokens = tokenize(normalized)

    reasons = []

    generic_count = sum(
        1 for token in tokens if token in GENERIC_NOUNS
    )

    vague_start = any(
        re.search(pattern, lowered)
        for pattern in VAGUE_START_PATTERNS
    )

    deictic_reference = any(
        re.search(pattern, lowered)
        for pattern in DEICTIC_PATTERNS
    )

    anchor = has_specific_anchor(normalized)

    if len(tokens) < 7 and generic_count >= 1 and not anchor:
        reasons.append(
            "very short question built around a generic subject"
        )

    if vague_start and generic_count >= 1 and not anchor:
        reasons.append(
            "generic wording does not identify the specific subject"
        )

    if deictic_reference and not anchor:
        reasons.append(
            "uses a context-dependent reference such as "
            "'the report' or 'the decision'"
        )

    # Examples such as "What were the two primary objectives of the review?"
    if (
        re.search(
            r"\b(?:two|three|primary|main|additional|specific)\b",
            lowered,
        )
        and generic_count >= 2
        and not anchor
    ):
        reasons.append(
            "asks for unnamed items from an unspecified source"
        )

    is_vague = bool(reasons)

    return {
        "flagged": is_vague,
        "reasons": list(dict.fromkeys(reasons)),
    }


# ============================================================
# LLM JUDGE
# ============================================================

JUDGE_PROMPT = """
You are auditing a failed retrieval result from a RAG evaluation.

Question:
{question}

Stored reference answer:
{reference}

Original gold chunk IDs:
{gold_chunk_ids}

Original gold passage(s):
{gold_context}

Retrieved chunk IDs and contexts:
{retrieved_context}

First evaluate QUESTION QUALITY independently of retrieval.

A question is too vague or ambiguous when a reader cannot reliably identify
the intended subject without already seeing the source passage. Examples:

- "What were the two primary objectives of the review?"
  Vague unless the question identifies which review.

- "What information is presented in Appendix V?"
  Vague unless the question identifies the report or document.

- "What difference did the draft report recognize?"
  Vague unless the question identifies the draft report and topic.

- "What did the decision say?"
  Vague unless the decision or case is named.

Do NOT mark a question vague merely because it is concise. A concise question
is acceptable when it contains a clear anchor such as an agency, named
program, law, report title, case, year, event, or distinctive technical term.

Then determine why retrieval recall was 0.

Use exactly one classification:
- TRUE_RETRIEVAL_FAILURE:
  None of the retrieved chunks independently contains enough evidence to
  answer the question correctly, and the question is sufficiently specific.
- INCOMPLETE_GOLD_LABEL:
  At least one retrieved chunk independently contains enough evidence to
  answer the question correctly, but that chunk ID is missing from the
  gold_chunk_ids list.
- AMBIGUOUS_QUESTION:
  The question is too vague, underspecified, or context dependent to be a
  reliable standalone evaluation item.
- BAD_REFERENCE_OR_GOLD_SPAN:
  The stored reference answer is incorrect, or the original gold passage
  does not fully support it.
- OTHER:
  Another issue explains the failure.

Return ONLY valid JSON with exactly these keys:
"classification",
"reason",
"vague_question" (true or false),
"vagueness_reason",
"suggested_rewrite",
"retrieved_answerable" (true or false),
"supporting_retrieved_chunk_ids" (list of integers),
"suggested_gold_chunk_ids" (list of integers).

Rules:
- If vague_question is true, classification should normally be
  AMBIGUOUS_QUESTION unless a more serious bad-reference problem exists.
- suggested_rewrite must make the intended subject explicit using only
  information available in the original gold passage.
- If no rewrite is needed, return an empty string.
- Only place a retrieved chunk ID in supporting_retrieved_chunk_ids if that
  chunk independently provides enough evidence to answer the question.
- suggested_gold_chunk_ids should contain the current gold IDs plus any
  retrieved chunks that independently support the answer.
- Do not add chunks merely because they are numerically adjacent or topically
  related.
- Keep each explanation concise.

Return JSON:
"""


def load_jsonl(path):
    records = []

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_number, line in enumerate(
            f,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Bad JSON on line {line_number} "
                    f"of {path}: {e}"
                ) from e

    return records


def format_retrieved_context(record):
    chunk_ids = record.get(
        "retrieved_chunk_ids"
    ) or []

    contexts = record.get(
        "retrieved_contexts"
    ) or []

    parts = []

    for rank, (chunk_id, context) in enumerate(
        zip(chunk_ids, contexts),
        start=1,
    ):
        parts.append(
            f"Rank {rank} | Chunk {chunk_id}\n"
            f"{context}"
        )

    return "\n\n---\n\n".join(parts)


def review_record(
    client,
    docs_by_id,
    record,
    sleep_seconds,
):
    question = record.get("question", "")

    fast_check = fast_vagueness_check(
        question
    )

    spans = record.get("gold_spans") or []

    if spans:
        gold_context = "\n\n---\n\n".join(
            span_text(docs_by_id, span)
            for span in spans
        )
    else:
        gold_context = (
            "[No gold spans supplied]"
        )

    prompt = JUDGE_PROMPT.format(
        question=question,
        reference=record.get(
            "reference",
            record.get("answer"),
        ),
        gold_chunk_ids=record.get(
            "gold_chunk_ids",
            [],
        ),
        gold_context=gold_context,
        retrieved_context=(
            format_retrieved_context(record)
        ),
    )

    try:
        result = call_gemini_json(
            client,
            prompt,
            sleep_seconds,
            max_retries=3,
        )
    except Exception as e:
        return {
            "classification": "ERROR",
            "reason": str(e),
            "vague_question": (
                fast_check["flagged"]
            ),
            "vagueness_reason": "; ".join(
                fast_check["reasons"]
            ),
            "suggested_rewrite": "",
            "fast_vagueness_flag": (
                fast_check["flagged"]
            ),
            "fast_vagueness_reasons": (
                fast_check["reasons"]
            ),
            "retrieved_answerable": None,
            "supporting_retrieved_chunk_ids": [],
            "suggested_gold_chunk_ids": (
                record.get(
                    "gold_chunk_ids",
                    [],
                )
            ),
        }

    valid_classes = {
        "TRUE_RETRIEVAL_FAILURE",
        "INCOMPLETE_GOLD_LABEL",
        "AMBIGUOUS_QUESTION",
        "BAD_REFERENCE_OR_GOLD_SPAN",
        "OTHER",
    }

    classification = result.get(
        "classification",
        "OTHER",
    )

    if classification not in valid_classes:
        classification = "OTHER"

    llm_vague = bool(
        result.get(
            "vague_question",
            False,
        )
    )

    # Use the LLM as the final decision, but preserve the fast
    # screen separately so disagreements can be reviewed.
    vague_question = llm_vague

    supporting_ids = [
        int(chunk_id)
        for chunk_id in result.get(
            "supporting_retrieved_chunk_ids",
            [],
        )
    ]

    suggested_ids = [
        int(chunk_id)
        for chunk_id in result.get(
            "suggested_gold_chunk_ids",
            record.get(
                "gold_chunk_ids",
                [],
            ),
        )
    ]

    return {
        "classification": classification,
        "reason": result.get(
            "reason",
            "",
        ),
        "vague_question": vague_question,
        "vagueness_reason": result.get(
            "vagueness_reason",
            "",
        ),
        "suggested_rewrite": result.get(
            "suggested_rewrite",
            "",
        ),
        "fast_vagueness_flag": (
            fast_check["flagged"]
        ),
        "fast_vagueness_reasons": (
            fast_check["reasons"]
        ),
        "vagueness_check_disagrees": (
            fast_check["flagged"]
            != llm_vague
        ),
        "retrieved_answerable": bool(
            result.get(
                "retrieved_answerable",
                False,
            )
        ),
        "supporting_retrieved_chunk_ids": (
            supporting_ids
        ),
        "suggested_gold_chunk_ids": (
            list(dict.fromkeys(
                suggested_ids
            ))
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path",
        default=str(DEFAULT_PATH),
    )

    parser.add_argument(
        "--dataset",
        default=str(
            DEFAULT_DATASET_PATH
        ),
    )

    parser.add_argument(
        "--out",
        default=str(
            DEFAULT_REPORT_PATH
        ),
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
    )

    args = parser.parse_args()

    input_path = Path(args.path)
    out_path = Path(args.out)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Failed-question file not found: "
            f"{input_path}\n"
            "Run extract_failed_questions.py "
            "first."
        )

    records = load_jsonl(
        input_path
    )

    print(
        f"Loaded {len(records)} failed "
        f"records from {input_path}"
    )

    docs = load_raw_docs(
        args.dataset
    )

    docs_by_id = {
        int(doc["doc_id"]): doc["text"]
        for doc in docs
    }

    client = get_gemini_client()

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    class_counts = {}
    vague_count = 0
    fast_vague_count = 0
    disagreement_count = 0

    with out_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for idx, record in enumerate(
            records,
            start=1,
        ):
            print(
                f"[{idx}/{len(records)}] "
                f"{record.get('question', '')[:80]}"
            )

            review = review_record(
                client,
                docs_by_id,
                record,
                args.sleep,
            )

            entry = dict(record)
            entry["failure_review"] = review

            f.write(
                json.dumps(
                    entry,
                    ensure_ascii=False,
                )
                + "\n"
            )

            f.flush()

            classification = review[
                "classification"
            ]

            class_counts[classification] = (
                class_counts.get(
                    classification,
                    0,
                )
                + 1
            )

            if review.get(
                "vague_question"
            ):
                vague_count += 1

            if review.get(
                "fast_vagueness_flag"
            ):
                fast_vague_count += 1

            if review.get(
                "vagueness_check_disagrees"
            ):
                disagreement_count += 1

            time.sleep(args.sleep)

    print()
    print("Failure review summary:")

    for classification, count in sorted(
        class_counts.items()
    ):
        print(
            f"  {classification:28s} "
            f"{count}"
        )

    print()
    print(
        f"LLM vague-question flags: "
        f"{vague_count}"
    )

    print(
        f"Fast vagueness flags: "
        f"{fast_vague_count}"
    )

    print(
        f"Fast/LLM disagreements: "
        f"{disagreement_count}"
    )

    print()
    print(
        f"Full report written to: "
        f"{out_path}"
    )


if __name__ == "__main__":
    main()