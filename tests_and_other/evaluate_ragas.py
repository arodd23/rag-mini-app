import asyncio
import csv
import json
import math
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from rag_core import RAGSystem
from metrics import recall_at_k, mrr, ndcg_at_k

from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
)


# ============================================================
# PATHS
# ============================================================

EVAL_PATH = Path("data/eval_set_v0.jsonl")

RESULTS_DIR = Path("results")

PREDICTIONS_PATH = (
    RESULTS_DIR / "evaluation_predictions.jsonl"
)

SCORECARD_PATH = (
    RESULTS_DIR / "evaluation_scorecard.csv"
)


# ============================================================
# SETTINGS
# ============================================================

# Set to 1 for the first test.
# After it works, set to None to run the full eval set.
MAX_QUESTIONS = 1

TOP_K = 5

MODEL_NAME = "gemini-2.5-flash"

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

MAX_OUTPUT_TOKENS = 8192

SECONDS_BETWEEN_API_STEPS = 15

SECONDS_BETWEEN_QUESTIONS = 30


# ============================================================
# API KEY
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


# ============================================================
# RAGAS MODEL SETUP
# ============================================================

def create_ragas_llm(api_key):
    """
    Use Gemini through Google's OpenAI-compatible endpoint.

    AsyncOpenAI is required because the Ragas collection
    metrics use async scoring.
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
        max_tokens=MAX_OUTPUT_TOKENS,
    )

    return ragas_llm


def create_ragas_embeddings():
    """
    Local Hugging Face embeddings.

    This avoids another hosted embedding API.

    AnswerRelevancy requires embeddings.
    """

    return HuggingFaceEmbeddings(
        model=EMBEDDING_MODEL
    )


# ============================================================
# LOAD EVAL SET
# ============================================================

def load_eval_set(
    path,
    max_questions=None,
):
    if not path.exists():
        raise FileNotFoundError(
            f"Eval file not found: {path}"
        )

    records = []

    with path.open(
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

def append_jsonl(
    path,
    record,
):
    with path.open(
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )

        f.flush()


def load_completed_records():
    """
    Load completed records.

    The newest result for an original_index wins.
    """

    completed = {}

    if not PREDICTIONS_PATH.exists():
        return completed

    with PREDICTIONS_PATH.open(
        "r",
        encoding="utf-8",
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

            original_index = record.get(
                "original_index"
            )

            if original_index is None:
                continue

            completed[original_index] = record

    return completed


# ============================================================
# SCORE HELPERS
# ============================================================

def get_score_value(result):
    if hasattr(result, "value"):
        value = result.value

    else:
        value = result

    value = float(value)

    if math.isnan(value):
        raise RuntimeError(
            "Ragas metric returned NaN."
        )

    return value


def get_score_reason(result):
    if hasattr(result, "reason"):
        return result.reason

    return None


# ============================================================
# EVALUATE ONE QUESTION
# ============================================================

async def evaluate_record(
    rag,
    eval_record,
    faithfulness_metric,
    answer_relevancy_metric,
):
    question = eval_record["question"]

    reference = eval_record["answer"]

    gold_chunk_id = int(
        eval_record["gold_chunk_id"]
    )

    # ========================================================
    # RETRIEVAL
    # ========================================================

    print()
    print("RETRIEVAL: retrieving top chunks")

    retrieved_chunks = rag.retrieve(
        question,
        top_k=TOP_K,
    )

    retrieved_chunk_ids = [
        chunk["chunk_id"]
        for chunk in retrieved_chunks
    ]

    retrieved_contexts = [
        chunk["text"]
        for chunk in retrieved_chunks
    ]

    print(
        f"Retrieved chunk IDs: "
        f"{retrieved_chunk_ids}"
    )

    print(
        f"Gold chunk ID: "
        f"{gold_chunk_id}"
    )

    recall_score = recall_at_k(
        retrieved_chunk_ids,
        gold_chunk_id,
        k=TOP_K,
    )

    mrr_score = mrr(
        retrieved_chunk_ids,
        gold_chunk_id,
    )

    ndcg_score = ndcg_at_k(
        retrieved_chunk_ids,
        gold_chunk_id,
        k=TOP_K,
    )

    print(
        f"Recall@{TOP_K}: "
        f"{recall_score}"
    )

    print(
        f"MRR: "
        f"{mrr_score}"
    )

    print(
        f"nDCG@{TOP_K}: "
        f"{ndcg_score}"
    )

    # ========================================================
    # RAG ANSWER
    # ========================================================

    print()
    print(
        "API CALL 1: "
        "RAG answer generation"
    )

    response, answer_retrieved_chunks = (
        rag.answer_question(
            question,
            top_k=TOP_K,
        )
    )

    response = response.strip()

    print()
    print(
        f"Generated response: "
        f"{response}"
    )

    print()
    print(
        f"Waiting "
        f"{SECONDS_BETWEEN_API_STEPS}s "
        f"before Ragas..."
    )

    await asyncio.sleep(
        SECONDS_BETWEEN_API_STEPS
    )

    # ========================================================
    # RAGAS FAITHFULNESS
    # ========================================================

    print()
    print(
        "RAGAS: Faithfulness"
    )

    faithfulness_result = (
        await faithfulness_metric.ascore(
            user_input=question,
            response=response,
            retrieved_contexts=(
                retrieved_contexts
            ),
        )
    )

    faithfulness_score = get_score_value(
        faithfulness_result
    )

    print(
        f"Faithfulness: "
        f"{faithfulness_score}"
    )

    await asyncio.sleep(
        SECONDS_BETWEEN_API_STEPS
    )

    # ========================================================
    # RAGAS ANSWER RELEVANCY
    # ========================================================

    print()
    print(
        "RAGAS: Answer Relevancy"
    )

    answer_relevancy_result = (
        await answer_relevancy_metric.ascore(
            user_input=question,
            response=response,
        )
    )

    answer_relevancy_score = get_score_value(
        answer_relevancy_result
    )

    print(
        f"Answer Relevancy: "
        f"{answer_relevancy_score}"
    )

    # ========================================================
    # FINAL RECORD
    # ========================================================

    return {
        "status": "complete",
        "original_index": (
            eval_record["original_index"]
        ),
        "question": question,
        "response": response,
        "reference": reference,
        "gold_chunk_id": gold_chunk_id,
        "retrieved_chunk_ids": (
            retrieved_chunk_ids
        ),
        "recall_at_5": recall_score,
        "mrr": mrr_score,
        "ndcg_at_5": ndcg_score,
        "faithfulness": (
            faithfulness_score
        ),
        "faithfulness_reason": (
            get_score_reason(
                faithfulness_result
            )
        ),
        "answer_relevancy": (
            answer_relevancy_score
        ),
        "answer_relevancy_reason": (
            get_score_reason(
                answer_relevancy_result
            )
        ),
    }


# ============================================================
# BUILD SCORECARD
# ============================================================

def build_scorecard(
    completed_records,
):
    if not completed_records:
        raise ValueError(
            "No completed evaluation records."
        )

    records = list(
        completed_records.values()
    )

    recall_scores = [
        float(record["recall_at_5"])
        for record in records
    ]

    mrr_scores = [
        float(record["mrr"])
        for record in records
    ]

    ndcg_scores = [
        float(record["ndcg_at_5"])
        for record in records
    ]

    faithfulness_scores = [
        float(record["faithfulness"])
        for record in records
    ]

    answer_relevancy_scores = [
        float(record["answer_relevancy"])
        for record in records
    ]

    scorecard = {
        "num_questions": len(records),

        "recall_at_5": (
            sum(recall_scores)
            / len(recall_scores)
        ),

        "mrr": (
            sum(mrr_scores)
            / len(mrr_scores)
        ),

        "ndcg_at_5": (
            sum(ndcg_scores)
            / len(ndcg_scores)
        ),

        "faithfulness": (
            sum(faithfulness_scores)
            / len(faithfulness_scores)
        ),

        "answer_relevancy": (
            sum(answer_relevancy_scores)
            / len(answer_relevancy_scores)
        ),
    }

    return scorecard


def write_scorecard(
    scorecard,
):
    fieldnames = [
        "num_questions",
        "recall_at_5",
        "mrr",
        "ndcg_at_5",
        "faithfulness",
        "answer_relevancy",
    ]

    with SCORECARD_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerow(
            scorecard
        )


# ============================================================
# PRINT SCORECARD
# ============================================================

def print_scorecard(
    scorecard,
):
    print()
    print("=" * 80)
    print(
        "PHASE 3 BASELINE SCORECARD"
    )
    print("=" * 80)

    print(
        f"Questions evaluated: "
        f"{scorecard['num_questions']}"
    )

    print()

    print(
        f"Recall@5: "
        f"{scorecard['recall_at_5']:.4f}"
    )

    print(
        f"MRR: "
        f"{scorecard['mrr']:.4f}"
    )

    print(
        f"nDCG@5: "
        f"{scorecard['ndcg_at_5']:.4f}"
    )

    print(
        f"Faithfulness: "
        f"{scorecard['faithfulness']:.4f}"
    )

    print(
        f"Answer Relevancy: "
        f"{scorecard['answer_relevancy']:.4f}"
    )

    print("=" * 80)


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

    completed_records = (
        load_completed_records()
    )

    print(
        f"Already completed records: "
        f"{len(completed_records)}"
    )

    # ========================================================
    # RAG SYSTEM
    # ========================================================

    print()
    print(
        "Initializing RAG system..."
    )

    rag = RAGSystem()

    rag.build_index()

    # ========================================================
    # RAGAS MODELS
    # ========================================================

    print()
    print(
        "Creating async Gemini "
        "Ragas evaluator..."
    )

    ragas_llm = create_ragas_llm(
        api_key
    )

    print(
        "Loading local embeddings "
        "for Answer Relevancy..."
    )

    ragas_embeddings = (
        create_ragas_embeddings()
    )

    # Ragas 0.4 collection metrics.
    faithfulness_metric = Faithfulness(
        llm=ragas_llm
    )

    answer_relevancy_metric = (
        AnswerRelevancy(
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )
    )

    print(
        "Ragas metrics ready."
    )

    # ========================================================
    # EVALUATION LOOP
    # ========================================================

    total = len(eval_records)

    for position, record in enumerate(
        eval_records,
        start=1,
    ):
        original_index = (
            record["original_index"]
        )

        if original_index in completed_records:
            print()
            print(
                f"Skipping completed record "
                f"{position}/{total}"
            )

            continue

        print()
        print("=" * 80)

        print(
            f"Processing record "
            f"{position}/{total}"
        )

        print(
            f"Question: "
            f"{record['question']}"
        )

        print("=" * 80)

        try:
            result_record = (
                await evaluate_record(
                    rag=rag,
                    eval_record=record,
                    faithfulness_metric=(
                        faithfulness_metric
                    ),
                    answer_relevancy_metric=(
                        answer_relevancy_metric
                    ),
                )
            )

            # Save immediately.
            append_jsonl(
                PREDICTIONS_PATH,
                result_record,
            )

            completed_records[
                original_index
            ] = result_record

            print()
            print("COMPLETED")

            print(
                f"Recall@5: "
                f"{result_record['recall_at_5']}"
            )

            print(
                f"MRR: "
                f"{result_record['mrr']}"
            )

            print(
                f"nDCG@5: "
                f"{result_record['ndcg_at_5']}"
            )

            print(
                f"Faithfulness: "
                f"{result_record['faithfulness']}"
            )

            print(
                f"Answer Relevancy: "
                f"{result_record['answer_relevancy']}"
            )

            # Rebuild scorecard after every question.
            scorecard = build_scorecard(
                completed_records
            )

            write_scorecard(
                scorecard
            )

            print(
                f"Checkpoint saved: "
                f"{PREDICTIONS_PATH}"
            )

            print(
                f"Scorecard updated: "
                f"{SCORECARD_PATH}"
            )

        except KeyboardInterrupt:
            print()
            print(
                "Stopped by user."
            )

            print(
                "Completed questions are saved."
            )

            print(
                "Rerun evaluate.py to resume."
            )

            raise

        except Exception as e:
            print()
            print("=" * 80)
            print("EVALUATION FAILED")
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

        if position < total:
            print()
            print(
                f"Waiting "
                f"{SECONDS_BETWEEN_QUESTIONS}s "
                f"before next question..."
            )

            await asyncio.sleep(
                SECONDS_BETWEEN_QUESTIONS
            )

    # ========================================================
    # FINAL SCORECARD
    # ========================================================

    if completed_records:
        scorecard = build_scorecard(
            completed_records
        )

        write_scorecard(
            scorecard
        )

        print_scorecard(
            scorecard
        )

    print()
    print(
        f"Predictions: "
        f"{PREDICTIONS_PATH}"
    )

    print(
        f"Scorecard: "
        f"{SCORECARD_PATH}"
    )


if __name__ == "__main__":
    asyncio.run(main())