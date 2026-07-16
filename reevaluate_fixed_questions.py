"""
reevaluate_fixed_questions.py

Reruns only corrected evaluation questions, replaces their latest cache
records, rebuilds evaluation_predictions.jsonl, and recalculates the full
scorecard using both:
    - untouched existing completed records
    - newly reevaluated corrected records

This is intended for questions that were rephrased or otherwise corrected
after the initial evaluation.

IMPORTANT:
Use the same original_index values as the master eval set. The script uses
original_index as the replacement key.

Expected fixed-question JSONL format:
{
  "original_index": 35,
  "question": "Rephrased question...",
  "answer": "Existing reference answer...",
  "type": "answerable",
  "gold_chunk_ids": [12345],
  "gold_spans": [{"doc_id": 123, "start": 1000, "end": 1500}]
}

For paraphrases, preserve the type and optional metadata:
{
  "original_index": 99,
  "question": "Paraphrased question...",
  "answer": "...",
  "type": "paraphrase",
  "gold_chunk_ids": [12345],
  "gold_spans": [...],
  "source_question": "..."
}

What the script does:
1. Loads the full master eval set.
2. Loads only corrected records from fixed_questions.jsonl.
3. Validates original_index and merges each correction with the master row.
4. Optionally patches the master eval set in place.
5. Loads the latest existing evaluation cache records.
6. Creates a fresh cache record for each corrected question.
7. Reruns retrieval, answer generation, faithfulness, and answer relevancy.
8. Replaces those indices in memory.
9. Compacts evaluation_cache.jsonl so each index appears only once.
10. Rebuilds evaluation_predictions.jsonl.
11. Recalculates and overwrites evaluation_scorecard.csv.

Usage:
    python reevaluate_fixed_questions.py

Custom paths:
    python reevaluate_fixed_questions.py \
        --fixed data/eval/fixed_questions.jsonl \
        --eval data/eval/eval_set_v2.jsonl \
        --cache data/cache/evaluation_cache.jsonl \
        --predictions results/evaluation_predictions.jsonl \
        --scorecard results/evaluation_scorecard.csv

Patch the master eval set with the corrected questions:
    python reevaluate_fixed_questions.py --update-eval-set

Reduce wait times for testing:
    python reevaluate_fixed_questions.py \
        --between-stages 2 \
        --between-questions 5
"""

import argparse
import asyncio
import json
import shutil
from datetime import datetime
from pathlib import Path

import evaluate


DEFAULT_FIXED_PATH = Path("data/eval/fixed_questions.jsonl")
DEFAULT_EVAL_PATH = Path("data/eval/eval_set_v2_raw.jsonl")
DEFAULT_CACHE_PATH = Path("data/cache/evaluation_cache.jsonl")
DEFAULT_PREDICTIONS_PATH = Path(
    "results/evaluation_predictions.jsonl"
)
DEFAULT_SCORECARD_PATH = Path(
    "results/evaluation_scorecard.csv"
)


# ============================================================
# FILE HELPERS
# ============================================================

def load_jsonl(path):
    records = []

    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Bad JSON on line {line_number} "
                    f"of {path}: {e}"
                ) from e

            records.append(record)

    return records


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(record, ensure_ascii=False)
                + "\n"
            )


def backup_file(path):
    path = Path(path)

    if not path.exists():
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(
        f"{path.name}.{timestamp}.bak"
    )

    shutil.copy2(path, backup_path)
    return backup_path


# ============================================================
# CORRECTION VALIDATION
# ============================================================

def normalize_fixed_record(fixed_record, master_record):
    """
    Merge a correction with its original master record.

    This allows fixed_questions.jsonl to include only:
        original_index
        question

    However, including all fields is recommended for clarity.
    """
    merged = dict(master_record)

    for key, value in fixed_record.items():
        if key == "reference":
            merged["answer"] = value
        else:
            merged[key] = value

    required = {
        "original_index",
        "question",
        "answer",
        "type",
        "gold_chunk_ids",
    }

    missing = required - set(merged)

    if missing:
        raise ValueError(
            f"Correction for index "
            f"{fixed_record.get('original_index')} "
            f"is missing fields after merge: "
            f"{sorted(missing)}"
        )

    if not isinstance(merged["original_index"], int):
        raise ValueError(
            "original_index must be an integer."
        )

    if not isinstance(merged["question"], str):
        raise ValueError(
            f"Question at index "
            f"{merged['original_index']} "
            "must be a string."
        )

    merged["question"] = merged["question"].strip()

    if not merged["question"]:
        raise ValueError(
            f"Question at index "
            f"{merged['original_index']} is empty."
        )

    if (
        merged["answer"] is not None
        and not isinstance(merged["answer"], str)
    ):
        raise ValueError(
            f"Answer at index "
            f"{merged['original_index']} "
            "must be a string or null."
        )

    if not isinstance(merged["gold_chunk_ids"], list):
        raise ValueError(
            f"gold_chunk_ids at index "
            f"{merged['original_index']} "
            "must be a list."
        )

    merged["gold_chunk_ids"] = list(dict.fromkeys(
        int(chunk_id)
        for chunk_id in merged["gold_chunk_ids"]
    ))

    if (
        merged["type"] == "unanswerable"
        and merged["gold_chunk_ids"]
    ):
        raise ValueError(
            f"Unanswerable record at index "
            f"{merged['original_index']} "
            "must have no gold chunks."
        )

    if (
        merged["type"] != "unanswerable"
        and not merged["gold_chunk_ids"]
    ):
        raise ValueError(
            f"Record at index "
            f"{merged['original_index']} "
            "must have at least one gold chunk."
        )

    return merged


def load_and_merge_corrections(
    fixed_path,
    master_records,
):
    fixed_records = load_jsonl(fixed_path)

    if not fixed_records:
        raise ValueError(
            f"No corrected records found in {fixed_path}"
        )

    master_by_index = {
        record["original_index"]: record
        for record in master_records
    }

    merged_records = []
    seen_indices = set()

    for fixed_record in fixed_records:
        original_index = fixed_record.get(
            "original_index"
        )

        if original_index is None:
            raise ValueError(
                "Every corrected record must include "
                "original_index."
            )

        if original_index in seen_indices:
            raise ValueError(
                f"Duplicate correction for "
                f"original_index={original_index}"
            )

        if original_index not in master_by_index:
            raise ValueError(
                f"Correction references unknown "
                f"original_index={original_index}. "
                f"Valid range is 0 to "
                f"{len(master_records) - 1}."
            )

        merged = normalize_fixed_record(
            fixed_record,
            master_by_index[original_index],
        )

        merged_records.append(merged)
        seen_indices.add(original_index)

    merged_records.sort(
        key=lambda record: record["original_index"]
    )

    return merged_records


# ============================================================
# MASTER EVAL UPDATE
# ============================================================

def patch_master_eval(
    eval_path,
    master_records,
    corrected_records,
):
    corrected_by_index = {
        record["original_index"]: record
        for record in corrected_records
    }

    updated_rows = []

    for master_record in master_records:
        original_index = master_record[
            "original_index"
        ]

        row = dict(
            corrected_by_index.get(
                original_index,
                master_record,
            )
        )

        # original_index is assigned by line order in evaluate.
        # It should not be written into the master eval file.
        row.pop("original_index", None)
        updated_rows.append(row)

    backup_path = backup_file(eval_path)
    write_jsonl(eval_path, updated_rows)

    print(
        f"Updated master eval set: {eval_path}"
    )

    if backup_path:
        print(
            f"Eval backup: {backup_path}"
        )


# ============================================================
# CACHE COMPACTION
# ============================================================

def compact_cache(cache_path, latest_records):
    """
    Physically replaces the append-only cache with one newest
    record per original_index.
    """
    cache_path = Path(cache_path)
    backup_path = backup_file(cache_path)

    ordered_records = [
        latest_records[index]
        for index in sorted(latest_records)
    ]

    write_jsonl(cache_path, ordered_records)

    print(
        f"Compacted cache to "
        f"{len(ordered_records)} latest records."
    )

    if backup_path:
        print(
            f"Cache backup: {backup_path}"
        )


# ============================================================
# TARGETED EVALUATION
# ============================================================

async def reevaluate_one(
    eval_record,
    rag,
    faithfulness_metric,
    answer_relevancy_metric,
    between_stages,
):
    """
    Always starts from a fresh cache record.

    Existing cached stages are intentionally ignored because
    the question or labels may have changed.
    """
    cache_record = evaluate.create_cache_record(
        eval_record
    )

    evaluate.save_cache_record(cache_record)

    cache_record = evaluate.run_retrieval_stage(
        rag=rag,
        cache_record=cache_record,
    )

    cache_record = evaluate.run_answer_stage(
        rag=rag,
        cache_record=cache_record,
    )

    if between_stages > 0:
        await asyncio.sleep(between_stages)

    cache_record = await evaluate.run_faithfulness_stage(
        cache_record=cache_record,
        faithfulness_metric=faithfulness_metric,
    )

    if between_stages > 0:
        await asyncio.sleep(between_stages)

    cache_record = (
        await evaluate.run_answer_relevancy_stage(
            cache_record=cache_record,
            answer_relevancy_metric=(
                answer_relevancy_metric
            ),
        )
    )

    return cache_record


# ============================================================
# MAIN
# ============================================================

async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fixed",
        default=str(DEFAULT_FIXED_PATH),
        help=(
            "JSONL containing only corrected records "
            "with original_index"
        ),
    )

    parser.add_argument(
        "--eval",
        default=str(DEFAULT_EVAL_PATH),
    )

    parser.add_argument(
        "--cache",
        default=str(DEFAULT_CACHE_PATH),
    )

    parser.add_argument(
        "--predictions",
        default=str(DEFAULT_PREDICTIONS_PATH),
    )

    parser.add_argument(
        "--scorecard",
        default=str(DEFAULT_SCORECARD_PATH),
    )

    parser.add_argument(
        "--update-eval-set",
        action="store_true",
        help=(
            "replace the corrected rows in the master "
            "eval JSONL before evaluation"
        ),
    )

    parser.add_argument(
        "--between-stages",
        type=float,
        default=evaluate.SECONDS_BETWEEN_API_STEPS,
    )

    parser.add_argument(
        "--between-questions",
        type=float,
        default=evaluate.SECONDS_BETWEEN_QUESTIONS,
    )

    args = parser.parse_args()

    fixed_path = Path(args.fixed)
    eval_path = Path(args.eval)
    cache_path = Path(args.cache)
    predictions_path = Path(args.predictions)
    scorecard_path = Path(args.scorecard)

    if not fixed_path.exists():
        raise FileNotFoundError(
            f"Corrected-question file not found: "
            f"{fixed_path}"
        )

    # Point imported evaluate helpers at the requested files.
    evaluate.EVAL_PATH = eval_path
    evaluate.CACHE_PATH = cache_path
    evaluate.PREDICTIONS_PATH = predictions_path
    evaluate.SCORECARD_PATH = scorecard_path
    evaluate.RESULTS_DIR = predictions_path.parent
    evaluate.CACHE_DIR = cache_path.parent

    predictions_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Loading master eval set: {eval_path}"
    )

    master_records = evaluate.load_eval_set(
        eval_path
    )

    corrected_records = (
        load_and_merge_corrections(
            fixed_path=fixed_path,
            master_records=master_records,
        )
    )

    print(
        f"Corrected questions to rerun: "
        f"{len(corrected_records)}"
    )

    for record in corrected_records:
        print(
            f"  index={record['original_index']} "
            f"type={record['type']} "
            f"question={record['question']}"
        )

    if args.update_eval_set:
        patch_master_eval(
            eval_path=eval_path,
            master_records=master_records,
            corrected_records=corrected_records,
        )

    # Existing completed records remain part of the final
    # scorecard unless their indices are reevaluated below.
    latest_records = evaluate.load_latest_records(
        cache_path
    )

    print(
        f"Existing latest cache records: "
        f"{len(latest_records)}"
    )

    api_key = evaluate.get_api_key()

    print()
    print("Initializing RAG system...")

    rag = evaluate.RAGSystem()
    rag.build_index()

    print()
    print("Creating Ragas evaluators...")

    ragas_llm = evaluate.create_ragas_llm(
        api_key
    )

    ragas_embeddings = (
        evaluate.create_ragas_embeddings()
    )

    faithfulness_metric = evaluate.Faithfulness(
        llm=ragas_llm
    )

    answer_relevancy_metric = (
        evaluate.AnswerRelevancy(
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )
    )

    total = len(corrected_records)

    for position, eval_record in enumerate(
        corrected_records,
        start=1,
    ):
        original_index = eval_record[
            "original_index"
        ]

        print()
        print("=" * 80)
        print(
            f"REEVALUATING "
            f"{position}/{total} "
            f"| original_index={original_index}"
        )
        print(
            f"Question: "
            f"{eval_record['question']}"
        )
        print("=" * 80)

        new_record = await reevaluate_one(
            eval_record=eval_record,
            rag=rag,
            faithfulness_metric=(
                faithfulness_metric
            ),
            answer_relevancy_metric=(
                answer_relevancy_metric
            ),
            between_stages=(
                args.between_stages
            ),
        )

        if new_record.get("status") != "complete":
            raise RuntimeError(
                f"Reevaluation did not complete for "
                f"original_index={original_index}"
            )

        # This is the actual logical replacement.
        latest_records[original_index] = (
            new_record
        )

        print()
        print(
            f"Replacement complete for index "
            f"{original_index}"
        )

        print(
            f"Recall@5: "
            f"{new_record['recall_at_5']}"
        )

        print(
            f"MRR: "
            f"{new_record['mrr']}"
        )

        print(
            f"nDCG@5: "
            f"{new_record['ndcg_at_5']}"
        )

        print(
            f"Faithfulness: "
            f"{new_record['faithfulness']}"
        )

        print(
            f"Answer Relevancy: "
            f"{new_record['answer_relevancy']}"
        )

        if (
            position < total
            and args.between_questions > 0
        ):
            await asyncio.sleep(
                args.between_questions
            )

    # Rewrite the append-only cache so old versions of corrected
    # records are physically removed.
    compact_cache(
        cache_path=cache_path,
        latest_records=latest_records,
    )

    # Rebuild final outputs from latest records only.
    evaluate.rebuild_predictions_file(
        latest_records
    )

    scorecard = evaluate.build_scorecard(
        latest_records
    )

    evaluate.write_scorecard(
        scorecard
    )

    evaluate.print_scorecard(
        scorecard
    )

    print()
    print(
        f"Updated cache: {cache_path}"
    )

    print(
        f"Updated predictions: "
        f"{predictions_path}"
    )

    print(
        f"Updated scorecard: "
        f"{scorecard_path}"
    )


if __name__ == "__main__":
    asyncio.run(main())