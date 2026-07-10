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

CACHE_DIR = Path("data/cache")

CACHE_PATH = (
    CACHE_DIR / "evaluation_cache.jsonl"
)

PREDICTIONS_PATH = (
    RESULTS_DIR / "evaluation_predictions.jsonl"
)

SCORECARD_PATH = (
    RESULTS_DIR / "evaluation_scorecard.csv"
)

# ============================================================
# SETTINGS
# ============================================================
MAX_QUESTIONS = None

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
# RAGAS SETUP
# ============================================================

def create_ragas_llm(api_key):
    """
    Gemini through Google's OpenAI-compatible endpoint.

    Ragas collection metrics use async scoring, so
    AsyncOpenAI is used here.
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

    return llm_factory(
        MODEL_NAME,
        provider="openai",
        client=async_client,
        adapter="instructor",
        max_tokens=MAX_OUTPUT_TOKENS,
    )


def create_ragas_embeddings():
    """
    Local embeddings for Answer Relevancy.
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
# JSON HELPERS
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


def load_latest_records(
    path,
):
    """
    Load JSONL records.

    If the same original_index appears multiple times,
    the newest record wins.
    """

    records = {}

    if not path.exists():
        return records

    with path.open(
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

            original_index = record.get(
                "original_index"
            )

            if original_index is None:
                continue

            records[original_index] = record

    return records


def save_cache_record(
    cache_record,
):
    """
    Append the newest state of a question.

    We intentionally append instead of rewriting the whole
    cache file. load_latest_records() keeps the newest state.
    """

    append_jsonl(
        CACHE_PATH,
        cache_record,
    )


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
# CREATE EMPTY CACHE RECORD
# ============================================================

def create_cache_record(
    eval_record,
):
    return {
        "original_index": (
            eval_record["original_index"]
        ),
        "question": (
            eval_record["question"]
        ),
        "reference": (
            eval_record["answer"]
        ),
        "gold_chunk_id": int(
            eval_record["gold_chunk_id"]
        ),

        "retrieved_chunk_ids": None,
        "retrieved_contexts": None,

        "recall_at_5": None,
        "mrr": None,
        "ndcg_at_5": None,

        "response": None,

        "faithfulness": None,
        "faithfulness_reason": None,

        "answer_relevancy": None,
        "answer_relevancy_reason": None,

        "status": "not_started",
    }


# ============================================================
# STAGE 1: RETRIEVAL
# ============================================================

def run_retrieval_stage(
    rag,
    cache_record,
):
    if (
        cache_record.get(
            "retrieved_chunk_ids"
        )
        is not None
        and cache_record.get(
            "retrieved_contexts"
        )
        is not None
        and cache_record.get(
            "recall_at_5"
        )
        is not None
        and cache_record.get(
            "mrr"
        )
        is not None
        and cache_record.get(
            "ndcg_at_5"
        )
        is not None
    ):
        print()
        print(
            "CACHE HIT: Retrieval already complete."
        )

        return cache_record

    print()
    print(
        "STAGE 1: Retrieval"
    )

    question = cache_record["question"]

    gold_chunk_id = cache_record[
        "gold_chunk_id"
    ]

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

    cache_record[
        "retrieved_chunk_ids"
    ] = retrieved_chunk_ids

    cache_record[
        "retrieved_contexts"
    ] = retrieved_contexts

    cache_record[
        "recall_at_5"
    ] = recall_score

    cache_record[
        "mrr"
    ] = mrr_score

    cache_record[
        "ndcg_at_5"
    ] = ndcg_score

    cache_record[
        "status"
    ] = "retrieval_complete"

    save_cache_record(
        cache_record
    )

    print(
        f"Retrieved chunk IDs: "
        f"{retrieved_chunk_ids}"
    )

    print(
        f"Gold chunk ID: "
        f"{gold_chunk_id}"
    )

    print(
        f"Recall@5: "
        f"{recall_score}"
    )

    print(
        f"MRR: "
        f"{mrr_score}"
    )

    print(
        f"nDCG@5: "
        f"{ndcg_score}"
    )

    print(
        "CACHE SAVED: Retrieval"
    )

    return cache_record


# ============================================================
# STAGE 2: RAG ANSWER
# ============================================================

def run_answer_stage(
    rag,
    cache_record,
):
    if cache_record.get("response"):
        print()
        print(
            "CACHE HIT: RAG answer already generated."
        )

        print(
            f"Cached response: "
            f"{cache_record['response']}"
        )

        return cache_record

    print()
    print(
        "STAGE 2: RAG answer generation"
    )

    print(
        "API CALL: RAG answer generation"
    )

    retrieved_chunks = [
    {
        "chunk_id": chunk_id,
        "text": chunk_text,
    }
    for chunk_id, chunk_text in zip(
        cache_record["retrieved_chunk_ids"],
        cache_record["retrieved_contexts"],
    )
    ]

    response = rag.answer_from_chunks(
    question=cache_record["question"],
    retrieved_chunks=retrieved_chunks,
    )

    cache_record[
        "response"
    ] = response

    cache_record[
        "status"
    ] = "answer_complete"

    save_cache_record(
        cache_record
    )

    print(
        f"Generated response: "
        f"{response}"
    )

    print(
        "CACHE SAVED: RAG answer"
    )

    return cache_record


# ============================================================
# STAGE 3: FAITHFULNESS
# ============================================================

async def run_faithfulness_stage(
    cache_record,
    faithfulness_metric,
):
    if (
        cache_record.get(
            "faithfulness"
        )
        is not None
    ):
        print()
        print(
            "CACHE HIT: Faithfulness already scored."
        )

        print(
            f"Cached Faithfulness: "
            f"{cache_record['faithfulness']}"
        )

        return cache_record

    print()
    print(
        "STAGE 3: Ragas Faithfulness"
    )

    print(
        "API STEP: Faithfulness"
    )

    faithfulness_result = (
        await faithfulness_metric.ascore(
            user_input=(
                cache_record["question"]
            ),
            response=(
                cache_record["response"]
            ),
            retrieved_contexts=(
                cache_record[
                    "retrieved_contexts"
                ]
            ),
        )
    )

    faithfulness_score = get_score_value(
        faithfulness_result
    )

    cache_record[
        "faithfulness"
    ] = faithfulness_score

    cache_record[
        "faithfulness_reason"
    ] = get_score_reason(
        faithfulness_result
    )

    cache_record[
        "status"
    ] = "faithfulness_complete"

    save_cache_record(
        cache_record
    )

    print(
        f"Faithfulness: "
        f"{faithfulness_score}"
    )

    print(
        "CACHE SAVED: Faithfulness"
    )

    return cache_record


# ============================================================
# STAGE 4: ANSWER RELEVANCY
# ============================================================

async def run_answer_relevancy_stage(
    cache_record,
    answer_relevancy_metric,
):
    if (
        cache_record.get(
            "answer_relevancy"
        )
        is not None
    ):
        print()
        print(
            "CACHE HIT: Answer Relevancy already scored."
        )

        print(
            f"Cached Answer Relevancy: "
            f"{cache_record['answer_relevancy']}"
        )

        return cache_record

    print()
    print(
        "STAGE 4: Ragas Answer Relevancy"
    )

    print(
        "API STEP: Answer Relevancy"
    )

    answer_relevancy_result = (
        await answer_relevancy_metric.ascore(
            user_input=(
                cache_record["question"]
            ),
            response=(
                cache_record["response"]
            ),
        )
    )

    answer_relevancy_score = get_score_value(
        answer_relevancy_result
    )

    cache_record[
        "answer_relevancy"
    ] = answer_relevancy_score

    cache_record[
        "answer_relevancy_reason"
    ] = get_score_reason(
        answer_relevancy_result
    )

    cache_record[
        "status"
    ] = "complete"

    save_cache_record(
        cache_record
    )

    print(
        f"Answer Relevancy: "
        f"{answer_relevancy_score}"
    )

    print(
        "CACHE SAVED: Answer Relevancy"
    )

    return cache_record


# ============================================================
# FINAL PREDICTION FILE
# ============================================================

def rebuild_predictions_file(
    cached_records,
):
    """
    Write only fully completed records to the official
    predictions file.
    """

    completed_records = [
        record
        for record in cached_records.values()
        if record.get("status") == "complete"
    ]

    completed_records.sort(
        key=lambda record: (
            record["original_index"]
        )
    )

    with PREDICTIONS_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        for record in completed_records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# SCORECARD
# ============================================================

def build_scorecard(
    cached_records,
):
    completed_records = [
        record
        for record in cached_records.values()
        if record.get("status") == "complete"
    ]

    if not completed_records:
        raise ValueError(
            "No completed evaluation records."
        )

    recall_scores = [
        float(record["recall_at_5"])
        for record in completed_records
    ]

    mrr_scores = [
        float(record["mrr"])
        for record in completed_records
    ]

    ndcg_scores = [
        float(record["ndcg_at_5"])
        for record in completed_records
    ]

    faithfulness_scores = [
        float(record["faithfulness"])
        for record in completed_records
    ]

    answer_relevancy_scores = [
        float(record["answer_relevancy"])
        for record in completed_records
    ]

    return {
        "num_questions": (
            len(completed_records)
        ),

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
    
    CACHE_DIR.mkdir(
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

    # ========================================================
    # CACHE
    # ========================================================

    cached_records = load_latest_records(
        CACHE_PATH
    )

    print(
        f"Cached question records: "
        f"{len(cached_records)}"
    )

    # ========================================================
    # RAG
    # ========================================================

    print()
    print(
        "Initializing RAG system..."
    )

    rag = RAGSystem()

    rag.build_index()

    # ========================================================
    # RAGAS
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

    for position, eval_record in enumerate(
        eval_records,
        start=1,
    ):
        original_index = (
            eval_record["original_index"]
        )

        print()
        print("=" * 80)

        print(
            f"Processing record "
            f"{position}/{total}"
        )

        print(
            f"Question: "
            f"{eval_record['question']}"
        )

        print("=" * 80)

        if original_index in cached_records:
            cache_record = cached_records[
                original_index
            ]

            print(
                f"Loaded cached status: "
                f"{cache_record.get('status')}"
            )

        else:
            cache_record = create_cache_record(
                eval_record
            )

            save_cache_record(
                cache_record
            )

            cached_records[
                original_index
            ] = cache_record

            print(
                "Created new cache record."
            )

        if (
            cache_record.get("status")
            == "complete"
        ):
            print()
            print(
                "CACHE HIT: Question already complete."
            )

            print(
                "Skipping all stages."
            )

            continue

        try:
            # =================================================
            # STAGE 1
            # =================================================

            cache_record = run_retrieval_stage(
                rag=rag,
                cache_record=cache_record,
            )

            cached_records[
                original_index
            ] = cache_record

            # =================================================
            # STAGE 2
            # =================================================

            cache_record = run_answer_stage(
                rag=rag,
                cache_record=cache_record,
            )

            cached_records[
                original_index
            ] = cache_record

            if (
                cache_record.get(
                    "faithfulness"
                )
                is None
            ):
                print()
                print(
                    f"Waiting "
                    f"{SECONDS_BETWEEN_API_STEPS}s "
                    f"before Faithfulness..."
                )

                await asyncio.sleep(
                    SECONDS_BETWEEN_API_STEPS
                )

            # =================================================
            # STAGE 3
            # =================================================

            cache_record = (
                await run_faithfulness_stage(
                    cache_record=cache_record,
                    faithfulness_metric=(
                        faithfulness_metric
                    ),
                )
            )

            cached_records[
                original_index
            ] = cache_record

            if (
                cache_record.get(
                    "answer_relevancy"
                )
                is None
            ):
                print()
                print(
                    f"Waiting "
                    f"{SECONDS_BETWEEN_API_STEPS}s "
                    f"before Answer Relevancy..."
                )

                await asyncio.sleep(
                    SECONDS_BETWEEN_API_STEPS
                )

            # =================================================
            # STAGE 4
            # =================================================

            cache_record = (
                await run_answer_relevancy_stage(
                    cache_record=cache_record,
                    answer_relevancy_metric=(
                        answer_relevancy_metric
                    ),
                )
            )

            cached_records[
                original_index
            ] = cache_record

            # =================================================
            # REBUILD OUTPUT FILES
            # =================================================

            rebuild_predictions_file(
                cached_records
            )

            scorecard = build_scorecard(
                cached_records
            )

            write_scorecard(
                scorecard
            )

            print()
            print("QUESTION COMPLETE")

            print(
                f"Recall@5: "
                f"{cache_record['recall_at_5']}"
            )

            print(
                f"MRR: "
                f"{cache_record['mrr']}"
            )

            print(
                f"nDCG@5: "
                f"{cache_record['ndcg_at_5']}"
            )

            print(
                f"Faithfulness: "
                f"{cache_record['faithfulness']}"
            )

            print(
                f"Answer Relevancy: "
                f"{cache_record['answer_relevancy']}"
            )

            print(
                f"Cache: "
                f"{CACHE_PATH}"
            )

            print(
                f"Predictions: "
                f"{PREDICTIONS_PATH}"
            )

            print(
                f"Scorecard: "
                f"{SCORECARD_PATH}"
            )

        except KeyboardInterrupt:
            print()
            print(
                "Stopped by user."
            )

            print(
                "All completed stages are cached."
            )

            print(
                "Rerun evaluate.py to resume "
                "from the missing stage."
            )

            raise

        except Exception as e:
            print()
            print("=" * 80)
            print(
                "EVALUATION STAGE FAILED"
            )
            print("=" * 80)

            print(
                f"Current cache status: "
                f"{cache_record.get('status')}"
            )

            print()

            print(
                f"Exception type: "
                f"{type(e).__name__}"
            )

            print()

            print(
                f"Exception message:\n{e}"
            )

            print()

            print(
                "Completed stages are already cached."
            )

            print(
                "Rerun evaluate.py and this question "
                "will resume from the missing stage."
            )

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
    # FINAL OUTPUT
    # ========================================================

    rebuild_predictions_file(
        cached_records
    )

    completed_count = sum(
        1
        for record in cached_records.values()
        if record.get("status") == "complete"
    )

    if completed_count > 0:
        scorecard = build_scorecard(
            cached_records
        )

        write_scorecard(
            scorecard
        )

        print_scorecard(
            scorecard
        )

    print()
    print(
        f"Stage cache: "
        f"{CACHE_PATH}"
    )

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