"""
review_eval_set.py

Automated first-pass QA on the generated eval set. This is NOT a
replacement for human review -- Phase 2/3 still call for you to eyeball a
sample and record a reject rate. This is a cheap filter to catch obviously
broken items first, and to give you an automated estimate of the reject
rate per question type before you spend human time on it.

Because every record carries gold_spans (durable doc_id/start/end offsets),
this script never touches chunk_index.jsonl -- only the raw docs and the
eval file. It works the same regardless of what chunking config was live
when the questions were generated.

Checks performed:
    answerable / multihop:
        LLM judge, given the gold span(s) as context: is the question
        answerable from just that context, and is the stored answer
        correct and supported? -> PASS / FAIL

    unanswerable:
        LLM judge, given the seed_span as context: can the question
        actually be answered from it? PASS means "genuinely not
        answerable" (correct); FAIL means the generator accidentally
        produced an answerable question.

    paraphrase:
        No LLM call. Embedding similarity (all-MiniLM-L6-v2) between the
        paraphrase and its source question. Flags if too similar (probably
        not a real paraphrase) or too dissimilar (meaning may have drifted).

Writes a full report (every record + verdict + reason) to a new JSONL file,
and prints a pass/fail/flag count per type as a first-pass reject-rate
estimate.

Usage:
    python review_eval_set.py
    python review_eval_set.py --path data/eval_set_v2_raw.jsonl
    python review_eval_set.py --skip-llm     # only run the free paraphrase check
"""

import argparse
import json
import time
from pathlib import Path

from corpus import load_raw_docs
from eval_common import get_gemini_client, call_gemini_json, span_text

DEFAULT_PATH = Path("data/eval/eval_set_v2_raw.jsonl")
DEFAULT_DATASET_PATH = Path("data/dataset/gov_report_sample_2k.json")
DEFAULT_REPORT_PATH = Path("data/results/eval_review_report.jsonl")
DEFAULT_SLEEP_SECONDS = 15

PARAPHRASE_MIN_SIM = 0.55  # below this: probably changed meaning
PARAPHRASE_MAX_SIM = 0.97  # above this: probably not really reworded

JUDGE_PROMPT_GROUNDED = """
You are auditing an evaluation item for a RAG system.

Context passage(s):
{context}

Question: {question}
Provided answer: {answer}

Check:
1. Is the question fully answerable using ONLY the context above?
2. Is the provided answer correct and fully supported by the context?
3. Is the question specific (not vague) and not a yes/no question?

Return ONLY valid JSON with exactly these keys:
"verdict" (either "PASS" or "FAIL"), "reason" (one short sentence).

Return JSON:
"""

JUDGE_PROMPT_UNANSWERABLE = """
You are auditing an "unanswerable" evaluation item for a RAG system. The
question below is supposed to NOT be answerable from the passage.

Passage:
{context}

Question: {question}

Check: can this question actually be fully or partially answered using
ONLY the passage above?

Return ONLY valid JSON with exactly these keys:
"verdict" ("PASS" if the question is genuinely NOT answerable from the
passage, "FAIL" if the passage actually does answer it),
"reason" (one short sentence).

Return JSON:
"""


def load_records(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def review_grounded(client, docs_by_id, record, sleep_seconds):
    spans = record.get("gold_spans", [])
    if not spans:
        return "FLAG", "no gold_spans on a grounded-type record"
    context = "\n\n---\n\n".join(span_text(docs_by_id, s) for s in spans)
    prompt = JUDGE_PROMPT_GROUNDED.format(
        context=context, question=record["question"], answer=record["answer"]
    )
    try:
        result = call_gemini_json(client, prompt, sleep_seconds, max_retries=3)
        return result.get("verdict", "FLAG"), result.get("reason", "")
    except Exception as e:
        return "ERROR", str(e)


def review_unanswerable(client, docs_by_id, record, sleep_seconds):
    seed_span = record.get("seed_span")
    if not seed_span:
        return "FLAG", "no seed_span on unanswerable record"
    context = span_text(docs_by_id, seed_span)
    prompt = JUDGE_PROMPT_UNANSWERABLE.format(context=context, question=record["question"])
    try:
        result = call_gemini_json(client, prompt, sleep_seconds, max_retries=3)
        return result.get("verdict", "FLAG"), result.get("reason", "")
    except Exception as e:
        return "ERROR", str(e)


def review_paraphrase(embedder, record):
    from sentence_transformers import util

    source_q = record.get("source_question")
    if not source_q:
        return "FLAG", "no source_question on paraphrase record", None

    emb = embedder.encode([source_q, record["question"]], normalize_embeddings=True)
    sim = float(util.cos_sim(emb[0], emb[1]))

    if sim > PARAPHRASE_MAX_SIM:
        return "FLAG", f"too similar to source (sim={sim:.2f}), may not be a real paraphrase", sim
    if sim < PARAPHRASE_MIN_SIM:
        return "FLAG", f"too dissimilar to source (sim={sim:.2f}), meaning may have changed", sim
    return "PASS", f"similarity={sim:.2f}", sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=str(DEFAULT_PATH))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--out", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="skip API calls entirely; only run the free paraphrase similarity check",
    )
    args = parser.parse_args()

    records = load_records(args.path)
    print(f"Loaded {len(records)} records from {args.path}")

    docs = load_raw_docs(args.dataset)
    docs_by_id = {d["doc_id"]: d["text"] for d in docs}

    client = None if args.skip_llm else get_gemini_client()

    embedder = None
    if any(r.get("type") == "paraphrase" for r in records):
        from sentence_transformers import SentenceTransformer
        print("Loading embedding model for paraphrase checks...")
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts = {}

    with out_path.open("w", encoding="utf-8") as f:
        for idx, record in enumerate(records, start=1):
            rtype = record.get("type", "unknown")
            print(f"[{idx}/{len(records)}] reviewing type={rtype}: {record.get('question', '')[:70]}")

            sim = None
            if rtype in ("answerable", "multihop"):
                if args.skip_llm:
                    verdict, reason = "SKIPPED", "LLM review skipped"
                else:
                    verdict, reason = review_grounded(client, docs_by_id, record, args.sleep)
                    time.sleep(args.sleep)
            elif rtype == "unanswerable":
                if args.skip_llm:
                    verdict, reason = "SKIPPED", "LLM review skipped"
                else:
                    verdict, reason = review_unanswerable(client, docs_by_id, record, args.sleep)
                    time.sleep(args.sleep)
            elif rtype == "paraphrase":
                verdict, reason, sim = review_paraphrase(embedder, record)
            else:
                verdict, reason = "FLAG", f"unknown type: {rtype}"

            entry = dict(record)
            entry["review_verdict"] = verdict
            entry["review_reason"] = reason
            if sim is not None:
                entry["review_similarity"] = sim

            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            key = (rtype, verdict)
            counts[key] = counts.get(key, 0) + 1

    print()
    print("Review summary:")
    for (rtype, verdict), n in sorted(counts.items()):
        print(f"  {rtype:14s} {verdict:8s} {n}")

    total_fail = sum(n for (_t, v), n in counts.items() if v == "FAIL")
    total_flag = sum(n for (_t, v), n in counts.items() if v == "FLAG")
    print()
    print(f"Total FAIL: {total_fail}  Total FLAG: {total_flag}  out of {len(records)}")
    print(f"Full report written to {out_path}")
    print("This is a first pass -- still do the human review the plan calls for.")


if __name__ == "__main__":
    main()