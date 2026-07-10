import asyncio
import csv
import json
import math
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from rag_core import RAGSystem

from ragas.llms import llm_factory
from ragas.metrics.collections import (
    ContextRecall,
    Faithfulness,
    FactualCorrectness,
)


# ============================================================
# PATHS
# ============================================================

EVAL_PATH = Path("data/eval_set_v0.jsonl")

RESULTS_DIR = Path("results")

RAGAS_INPUT_PATH = (
    RESULTS_DIR / "ragas2_input_checkpoint.jsonl"
)

RAGAS_RESULTS_JSONL_PATH = (
    RESULTS_DIR / "ragas2_results_checkpoint.jsonl"
)

RAGAS_SCORES_CSV_PATH = (
    RESULTS_DIR / "ragas2_scores_checkpoint.csv"
)


# ============================================================
# SETTINGS
# ============================================================

# KEEP AT 1 UNTIL THIS TEST WORKS
MAX_QUESTIONS = 1

TOP_K = 5

MODEL_NAME = "gemini-2.5-flash"

SECONDS_BETWEEN_API_STEPS = 15

SECONDS_BETWEEN_QUESTIONS = 30

# IMPORTANT:
# Give Ragas enough room for claim decomposition / verification.
MAX_OUTPUT_TOKENS = 8192


# ============================================================
# API SETUP
# ============================================================

def get_api_key():
    load_dotenv(override=True)

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "Missing GEMINI_API_KEY in .env file."
        )

    print(
        f"Using GEMINI_API_KEY ending in "
        f"...{api_key[-6:]}"
    )

    return api_key


def create_ragas_llm(api_key):
    """
    Gemini through Google's OpenAI-compatible endpoint.

    AsyncOpenAI is used because modern Ragas collection
    metrics call agenerate() through ascore().
    """

    async_client = AsyncOpenAI(
        api_key=api_key,
        base_url=(
            "https://generativelanguage.googleapis.com/"
            "v1beta/openai/"
        ),
        max_retries=1,
        timeout=300.0,
    )

    ragas_llm = llm_factory(
        MODEL_NAME,
        provider="openai",
        client=async_client,
        adapter="instructor",

        # IMPORTANT FIX:
        # This is passed to model generation calls.
        max_tokens=MAX_OUTPUT_TOKENS,
    )

    return ragas_llm


# ============================================================
# LOAD EVAL DATA
# ============================================================

def load_eval_set(path, max_questions=None):
    if not path.exists():
        raise FileNotFoundError(
            f"Eval file not found: {path}"
        )

    records = []

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1
        ):
            if not line.strip():
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Bad JSON on line "
                    f"{line_number}: {e}"
                ) from e

            required_keys = {
                "question",
                "answer",
                "gold_chunk_id",
            }

            if not required_keys.issubset(
                record.keys()
            ):
                raise ValueError(
                    f"Bad record on line "
                    f"{line_number}. "
                    f"Missing required keys."
                )

            record["original_index"] = len(records)

            records.append(record)

            if (
                max_questions is not None
                and len(records) >= max_questions
            ):
                break

    return records


# ============================================================
# JSONL HELPERS
# ============================================================

def append_jsonl(path, record):
    with path.open(
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )

        f.flush()


def load_completed_indices():
    completed = set()

    if not RAGAS_RESULTS_JSONL_PATH.exists():
        return completed

    with RAGAS_RESULTS_JSONL_PATH.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            if not line.strip():
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError:
                continue

            if record.get("status") == "complete":
                completed.add(
                    record["original_index"]
                )

    return completed


# ============================================================
# RAG ANSWER GENERATION
# ============================================================

def build_ragas_record(
    rag,
    eval_record,
):
    print(
        "API CALL 1: "
        "RAG answer generation"
    )

    response, retrieved_chunks = (
        rag.answer_question(
            eval_record["question"],
            top_k=TOP_K,
        )
    )

    retrieved_contexts = [
        chunk["text"]
        for chunk in retrieved_chunks
    ]

    retrieved_chunk_ids = [
        chunk["chunk_id"]
        for chunk in retrieved_chunks
    ]

    ragas_record = {
        "original_index": (
            eval_record["original_index"]
        ),
        "user_input": (
            eval_record["question"]
        ),
        "retrieved_contexts": (
            retrieved_contexts
        ),
        "retrieved_chunk_ids": (
            retrieved_chunk_ids
        ),
        "response": response.strip(),
        "reference": (
            eval_record["answer"]
        ),
        "gold_chunk_id": (
            eval_record["gold_chunk_id"]
        ),
    }

    append_jsonl(
        RAGAS_INPUT_PATH,
        ragas_record,
    )

    return ragas_record


# ============================================================
# RESULT HELPERS
# ============================================================

def get_score_value(result):
    if hasattr(result, "value"):
        value = result.value
    else:
        value = result

    value = float(value)

    if math.isnan(value):
        raise RuntimeError(
            "Metric returned NaN."
        )

    return value


def get_score_reason(result):
    if hasattr(result, "reason"):
        return result.reason

    return None


# ============================================================
# OFFICIAL RAGAS METRICS
# ============================================================

async def run_ragas_metrics(
    ragas_record,
    ragas_llm,
):
    print()
    print(
        "Creating official RAGAS scorers..."
    )

    context_recall_metric = ContextRecall(
        llm=ragas_llm
    )

    faithfulness_metric = Faithfulness(
        llm=ragas_llm
    )

    factual_correctness_metric = (
        FactualCorrectness(
            llm=ragas_llm
        )
    )

    # ========================================================
    # CONTEXT RECALL
    # ========================================================

    print()
    print(
        "API STEP 2: "
        "RAGAS ContextRecall"
    )

    context_recall_result = (
        await context_recall_metric.ascore(
            user_input=(
                ragas_record["user_input"]
            ),
            retrieved_contexts=(
                ragas_record[
                    "retrieved_contexts"
                ]
            ),
            reference=(
                ragas_record["reference"]
            ),
        )
    )

    context_recall = get_score_value(
        context_recall_result
    )

    print(
        f"ContextRecall: "
        f"{context_recall}"
    )

    await asyncio.sleep(
        SECONDS_BETWEEN_API_STEPS
    )

    # ========================================================
    # FAITHFULNESS
    # ========================================================

    print()
    print(
        "API STEP 3: "
        "RAGAS Faithfulness"
    )

    faithfulness_result = (
        await faithfulness_metric.ascore(
            user_input=(
                ragas_record["user_input"]
            ),
            response=(
                ragas_record["response"]
            ),
            retrieved_contexts=(
                ragas_record[
                    "retrieved_contexts"
                ]
            ),
        )
    )

    faithfulness = get_score_value(
        faithfulness_result
    )

    print(
        f"Faithfulness: "
        f"{faithfulness}"
    )

    await asyncio.sleep(
        SECONDS_BETWEEN_API_STEPS
    )

    # ========================================================
    # FACTUAL CORRECTNESS
    # ========================================================

    print()
    print(
        "API STEP 4: "
        "RAGAS FactualCorrectness"
    )

    factual_correctness_result = (
        await factual_correctness_metric.ascore(
            response=(
                ragas_record["response"]
            ),
            reference=(
                ragas_record["reference"]
            ),
        )
    )

    factual_correctness = get_score_value(
        factual_correctness_result
    )

    print(
        f"FactualCorrectness: "
        f"{factual_correctness}"
    )

    return {
        "status": "complete",

        "original_index": (
            ragas_record["original_index"]
        ),

        "user_input": (
            ragas_record["user_input"]
        ),

        "retrieved_contexts": (
            ragas_record[
                "retrieved_contexts"
            ]
        ),

        "retrieved_chunk_ids": (
            ragas_record[
                "retrieved_chunk_ids"
            ]
        ),

        "response": (
            ragas_record["response"]
        ),

        "reference": (
            ragas_record["reference"]
        ),

        "gold_chunk_id": (
            ragas_record["gold_chunk_id"]
        ),

        "context_recall": (
            context_recall
        ),

        "context_recall_reason": (
            get_score_reason(
                context_recall_result
            )
        ),

        "faithfulness": (
            faithfulness
        ),

        "faithfulness_reason": (
            get_score_reason(
                faithfulness_result
            )
        ),

        "factual_correctness": (
            factual_correctness
        ),

        "factual_correctness_reason": (
            get_score_reason(
                factual_correctness_result
            )
        ),
    }


# ============================================================
# CSV OUTPUT
# ============================================================

def rebuild_csv_from_jsonl():
    if not RAGAS_RESULTS_JSONL_PATH.exists():
        return

    completed_by_index = {}

    with RAGAS_RESULTS_JSONL_PATH.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            if not line.strip():
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError:
                continue

            if record.get("status") != "complete":
                continue

            completed_by_index[
                record["original_index"]
            ] = record

    rows = list(
        completed_by_index.values()
    )

    rows.sort(
        key=lambda x: x["original_index"]
    )

    fieldnames = [
        "original_index",
        "user_input",
        "retrieved_chunk_ids",
        "response",
        "reference",
        "gold_chunk_id",
        "context_recall",
        "faithfulness",
        "factual_correctness",
    ]

    with RAGAS_SCORES_CSV_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow({
                key: row.get(key)
                for key in fieldnames
            })

    print(
        f"CSV updated: "
        f"{RAGAS_SCORES_CSV_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

async def main():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    api_key = get_api_key()

    print(
        f"Loading eval set from: "
        f"{EVAL_PATH}"
    )

    eval_records = load_eval_set(
        EVAL_PATH,
        max_questions=MAX_QUESTIONS,
    )

    print(
        f"Loaded records: "
        f"{len(eval_records)}"
    )

    completed_indices = (
        load_completed_indices()
    )

    print(
        f"Already completed records: "
        f"{len(completed_indices)}"
    )

    print()
    print(
        "Initializing RAG system..."
    )

    rag = RAGSystem()

    rag.build_index()

    print()
    print(
        "Creating async Gemini evaluator..."
    )

    ragas_llm = create_ragas_llm(
        api_key
    )

    print(
        "RAGAS evaluator ready."
    )

    print(
        f"Evaluator max output tokens: "
        f"{MAX_OUTPUT_TOKENS}"
    )

    total = len(eval_records)

    for record in eval_records:
        original_index = (
            record["original_index"]
        )

        if original_index in completed_indices:
            print(
                f"Skipping completed record "
                f"{original_index + 1}/{total}"
            )

            continue

        print()
        print("=" * 80)

        print(
            f"Processing record "
            f"{original_index + 1}/{total}"
        )

        print(
            f"Question: "
            f"{record['question']}"
        )

        print("=" * 80)

        try:
            ragas_record = (
                build_ragas_record(
                    rag=rag,
                    eval_record=record,
                )
            )

            print()
            print(
                f"Waiting "
                f"{SECONDS_BETWEEN_API_STEPS}s "
                f"before RAGAS..."
            )

            await asyncio.sleep(
                SECONDS_BETWEEN_API_STEPS
            )

            result_record = (
                await run_ragas_metrics(
                    ragas_record=ragas_record,
                    ragas_llm=ragas_llm,
                )
            )

            append_jsonl(
                RAGAS_RESULTS_JSONL_PATH,
                result_record,
            )

            completed_indices.add(
                original_index
            )

            rebuild_csv_from_jsonl()

            print()
            print("COMPLETED")

            print(
                "context_recall: "
                f"{result_record['context_recall']}"
            )

            print(
                "faithfulness: "
                f"{result_record['faithfulness']}"
            )

            print(
                "factual_correctness: "
                f"{result_record['factual_correctness']}"
            )

        except KeyboardInterrupt:
            print()
            print(
                "Stopped by user."
            )

            print(
                "Completed questions are saved."
            )

            raise

        except Exception as e:
            print()
            print("=" * 80)
            print("FAILED")
            print("=" * 80)

            print(
                f"Exception type: "
                f"{type(e).__name__}"
            )

            print()

            print(
                f"Exception message:\n{e}"
            )

            print()

            raise

        if (
            original_index
            != eval_records[-1]["original_index"]
        ):
            print()
            print(
                f"Waiting "
                f"{SECONDS_BETWEEN_QUESTIONS}s "
                f"before next question..."
            )

            await asyncio.sleep(
                SECONDS_BETWEEN_QUESTIONS
            )

    print()
    print("=" * 80)
    print(
        "RAGAS EVALUATION COMPLETE"
    )
    print("=" * 80)

    rebuild_csv_from_jsonl()

    print(
        f"Final CSV: "
        f"{RAGAS_SCORES_CSV_PATH}"
    )


if __name__ == "__main__":
    asyncio.run(main())