"""
evaluate.py

Phase 3 evaluation harness, rebuilt from your version with two fixes for
Phase 4:

1. Retrieval scoring now uses gold_spans (character-offset overlap), not
   gold_chunk_ids (exact chunk-id match). Chunk ids are only meaningful
   within the ONE chunk_index that produced them -- the moment you sweep a
   different chunk_size/overlap/strategy, old chunk ids point at different
   text. gold_spans are durable across any chunking config.

2. Everything is config-aware. The chunking config (strategy/size/overlap/
   similarity_threshold) is a CLI argument. The per-question stage cache
   and predictions file are namespaced by config name, so two different
   configs never share (or silently corrupt) each other's cached retrieval/
   answer/faithfulness/relevancy results. The scorecard CSV is the one
   thing that stays cumulative -- each run APPENDS a row (with config
   columns) so a sweep naturally builds the comparison table Phase 4 wants.

Same stage-based resume design as before: each question moves through
retrieval -> answer -> faithfulness -> answer_relevancy, cached after each
stage, so a killed run resumes from wherever it left off. Same real Ragas
wiring (Gemini via its OpenAI-compatible endpoint + local HF embeddings).

Usage:
    python evaluate.py                                  # default config: fixed, 500/50
    python evaluate.py --strategy recursive --chunk-size 1024 --overlap 100
    python evaluate.py --strategy semantic --chunk-size 1024 --similarity-threshold 0.6
    python evaluate.py --max-questions 40               # subset, for sweeps
    python evaluate.py --skip-ragas                      # retrieval-only smoke test
"""

import argparse
import asyncio
import csv
import json
import math
import os
import requests
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from corpus import ChunkConfig, file_hash
from rag_core import RAGSystem
from metrics import recall_at_k_multi, mrr_multi, ndcg_at_k_multi

from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.metrics.collections import Faithfulness, AnswerRelevancy


# ============================================================
# PATHS
# ============================================================

EVAL_PATH = Path("data/eval/eval_set_v2_raw.jsonl")
RESULTS_DIR = Path("results")
CACHE_DIR = Path("data/cache")

# Cumulative across every config -- one row appended per evaluate.py run.
SCORECARD_PATH = RESULTS_DIR / "evaluation_scorecard.csv"

SCORECARD_FIELDS = [
    "config_name", "strategy", "chunk_size", "overlap", "similarity_threshold",
    "top_k", "eval_file_hash", "dataset_hash",
    "num_questions", "num_unanswerable",
    "num_scored_for_retrieval", "num_scored_for_faithfulness", "num_scored_for_answer_relevancy",
    "recall_at_5", "mrr", "ndcg_at_5", "faithfulness", "answer_relevancy",
]


def cache_path_for(config_name):
    return CACHE_DIR / f"evaluation_cache__{config_name}.jsonl"


def predictions_path_for(config_name):
    return RESULTS_DIR / f"evaluation_predictions__{config_name}.jsonl"


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "gemini-2.5-flash"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_OUTPUT_TOKENS = 8192
SECONDS_BETWEEN_API_STEPS = 15
SECONDS_BETWEEN_QUESTIONS = 30

# ============================================================
# Notification
# ============================================================

def send_phone_notification(title: str, message: str) -> None:
    topic = "aditirod_thirdy"

    try:
        response = requests.post(
            f"https://ntfy.sh/{topic}",
            data=message.encode("utf-8"),
            headers={
                "Title": title,
                "Priority": "high",
                "Tags": "white_check_mark",
            },
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Phone notification failed: {exc}")

# ============================================================
# API KEY
# ============================================================

def get_api_key():
    load_dotenv(override=True)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY in .env file.")
    print(f"Using GEMINI_API_KEY ending in ...{api_key[-6:]}")
    return api_key


# ============================================================
# RAGAS SETUP
# ============================================================

def create_ragas_llm(api_key):
    """Gemini through Google's OpenAI-compatible endpoint. Ragas collection
    metrics use async scoring, so AsyncOpenAI is used here."""
    async_client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
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
    """Local embeddings for Answer Relevancy."""
    return HuggingFaceEmbeddings(model=EMBEDDING_MODEL)


# ============================================================
# LOAD EVAL SET
# ============================================================

def load_eval_set(path, max_questions=None):
    if not path.exists():
        raise FileNotFoundError(f"Eval file not found: {path}")

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON on line {line_number}: {e}") from e

            required_keys = {"question", "answer", "type", "gold_chunk_ids", "gold_spans"}
            if not required_keys.issubset(record.keys()):
                raise ValueError(f"Bad record on line {line_number}. Missing required keys.")

            if not isinstance(record["question"], str) or not record["question"].strip():
                raise ValueError(f"Bad record on line {line_number}: question must be non-empty.")
            if not isinstance(record["type"], str) or not record["type"].strip():
                raise ValueError(f"Bad record on line {line_number}: type must be non-empty.")
            if record["answer"] is not None and not isinstance(record["answer"], str):
                raise ValueError(f"Bad record on line {line_number}: answer must be a string or null.")

            gold_ids = record["gold_chunk_ids"] or []
            if not isinstance(gold_ids, list):
                raise ValueError(f"Bad record on line {line_number}: gold_chunk_ids must be a list.")
            record["gold_chunk_ids"] = list(dict.fromkeys(int(g) for g in gold_ids))

            gold_spans = record["gold_spans"] or []
            if not isinstance(gold_spans, list):
                raise ValueError(f"Bad record on line {line_number}: gold_spans must be a list.")
            record["gold_spans"] = gold_spans

            if record["type"] == "unanswerable":
                if record["gold_chunk_ids"] or record["gold_spans"]:
                    raise ValueError(f"Bad record on line {line_number}: unanswerable items must have no gold evidence.")
            else:
                if not record["gold_chunk_ids"] or not record["gold_spans"]:
                    raise ValueError(f"Bad record on line {line_number}: answerable items need gold_chunk_ids and gold_spans.")

            record["original_index"] = len(records)
            records.append(record)

            if max_questions is not None and len(records) >= max_questions:
                break

    return records


# ============================================================
# JSON / CACHE HELPERS
# ============================================================

def append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def load_latest_records(path):
    """Load JSONL records; if original_index repeats, the newest wins."""
    records = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            original_index = record.get("original_index")
            if original_index is not None:
                records[original_index] = record
    return records


def save_cache_record(cache_record, cache_path):
    """Append the newest state of a question. load_latest_records() keeps the newest."""
    append_jsonl(cache_path, cache_record)


# ============================================================
# SCORE HELPERS
# ============================================================

def get_score_value(result):
    """Ragas returns NaN when there's nothing to measure (e.g. a refusal
    with zero verifiable claims -> Faithfulness computes 0/0). Treated as
    None ("not applicable"), not a crash."""
    value = result.value if hasattr(result, "value") else result
    value = float(value)
    return None if math.isnan(value) else value


def get_score_reason(result):
    return result.reason if hasattr(result, "reason") else None


# ============================================================
# CACHE RECORD
# ============================================================

def create_cache_record(eval_record, config_name):
    return {
        "original_index": eval_record["original_index"],
        "question": eval_record["question"],
        "reference": eval_record["answer"],
        "type": eval_record["type"],
        "gold_chunk_ids": eval_record["gold_chunk_ids"],
        "gold_spans": eval_record["gold_spans"],
        "config_name": config_name,

        "retrieved_chunk_ids": None,
        "retrieved_doc_ids": None,
        "retrieved_contexts": None,
        "retrieved_spans": None,

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
# STAGE 1: RETRIEVAL  (fixed: scores against gold_spans, not gold_chunk_ids)
# ============================================================

def run_retrieval_stage(rag, cache_record, cache_path, top_k):
    if cache_record.get("retrieved_chunk_ids") is not None:
        print("\nCACHE HIT: Retrieval already complete.")
        return cache_record

    print("\nSTAGE 1: Retrieval")

    question = cache_record["question"]
    gold_spans = cache_record["gold_spans"]

    retrieved_chunks = rag.retrieve(question, top_k=top_k)

    # retrieved_chunks already carry doc_id/start/end -- that IS the span
    # shape metrics.py expects. No conversion needed.
    recall_score = recall_at_k_multi(retrieved_chunks, gold_spans, k=top_k)
    mrr_score = mrr_multi(retrieved_chunks, gold_spans)
    ndcg_score = ndcg_at_k_multi(retrieved_chunks, gold_spans, k=top_k)

    cache_record["retrieved_chunk_ids"] = [c["chunk_id"] for c in retrieved_chunks]
    cache_record["retrieved_doc_ids"] = [c["doc_id"] for c in retrieved_chunks]
    cache_record["retrieved_contexts"] = [c["text"] for c in retrieved_chunks]
    cache_record["retrieved_spans"] = [
        {"doc_id": c["doc_id"], "start": c["start"], "end": c["end"]} for c in retrieved_chunks
    ]
    cache_record["recall_at_5"] = recall_score
    cache_record["mrr"] = mrr_score
    cache_record["ndcg_at_5"] = ndcg_score
    cache_record["status"] = "retrieval_complete"

    save_cache_record(cache_record, cache_path)

    print(f"Retrieved chunk IDs: {cache_record['retrieved_chunk_ids']}")
    print(f"Gold span(s): {gold_spans}")
    print(f"Recall@5: {recall_score}")
    print(f"MRR: {mrr_score}")
    print(f"nDCG@5: {ndcg_score}")
    print("CACHE SAVED: Retrieval")

    return cache_record


# ============================================================
# STAGE 2: RAG ANSWER
# ============================================================

def run_answer_stage(rag, cache_record, cache_path):
    if cache_record.get("response"):
        print("\nCACHE HIT: RAG answer already generated.")
        print(f"Cached response: {cache_record['response']}")
        return cache_record

    print("\nSTAGE 2: RAG answer generation")
    print("API CALL: RAG answer generation")

    retrieved_chunks = [
        {"chunk_id": chunk_id, "text": chunk_text}
        for chunk_id, chunk_text in zip(cache_record["retrieved_chunk_ids"], cache_record["retrieved_contexts"])
    ]

    response = rag.answer_from_chunks(question=cache_record["question"], retrieved_chunks=retrieved_chunks)

    cache_record["response"] = response
    cache_record["status"] = "answer_complete"
    save_cache_record(cache_record, cache_path)

    print(f"Generated response: {response}")
    print("CACHE SAVED: RAG answer")

    return cache_record


# ============================================================
# STAGE 3: FAITHFULNESS
# ============================================================

async def run_faithfulness_stage(cache_record, cache_path, faithfulness_metric):
    if cache_record.get("status") in {"faithfulness_complete", "complete"}:
        print("\nCACHE HIT: Faithfulness already scored.")
        print(f"Cached Faithfulness: {cache_record['faithfulness']}")
        return cache_record

    print("\nSTAGE 3: Ragas Faithfulness")
    print("API STEP: Faithfulness")

    result = await faithfulness_metric.ascore(
        user_input=cache_record["question"],
        response=cache_record["response"],
        retrieved_contexts=cache_record["retrieved_contexts"],
    )

    faithfulness_score = get_score_value(result)
    cache_record["faithfulness"] = faithfulness_score
    cache_record["faithfulness_reason"] = get_score_reason(result)
    cache_record["status"] = "faithfulness_complete"
    save_cache_record(cache_record, cache_path)

    if faithfulness_score is None:
        print("NOTE: Faithfulness came back NaN (commonly a refusal with no "
              "verifiable claims). Recorded as N/A, not a failure.")

    print(f"Faithfulness: {faithfulness_score}")
    print("CACHE SAVED: Faithfulness")

    return cache_record


# ============================================================
# STAGE 4: ANSWER RELEVANCY
# ============================================================

async def run_answer_relevancy_stage(cache_record, cache_path, answer_relevancy_metric):
    if cache_record.get("status") == "complete":
        print("\nCACHE HIT: Answer Relevancy already scored.")
        print(f"Cached Answer Relevancy: {cache_record['answer_relevancy']}")
        return cache_record

    print("\nSTAGE 4: Ragas Answer Relevancy")
    print("API STEP: Answer Relevancy")

    result = await answer_relevancy_metric.ascore(
        user_input=cache_record["question"],
        response=cache_record["response"],
    )

    relevancy_score = get_score_value(result)
    cache_record["answer_relevancy"] = relevancy_score
    cache_record["answer_relevancy_reason"] = get_score_reason(result)
    cache_record["status"] = "complete"
    save_cache_record(cache_record, cache_path)

    if relevancy_score is None:
        print("NOTE: Answer Relevancy came back NaN. Recorded as N/A, not a failure.")

    print(f"Answer Relevancy: {relevancy_score}")
    print("CACHE SAVED: Answer Relevancy")

    return cache_record


# ============================================================
# PREDICTIONS FILE
# ============================================================

def rebuild_predictions_file(cached_records, predictions_path):
    completed = [r for r in cached_records.values() if r.get("status") == "complete"]
    completed.sort(key=lambda r: r["original_index"])
    with predictions_path.open("w", encoding="utf-8") as f:
        for record in completed:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ============================================================
# SCORECARD  (now: config-tagged row, APPENDED to a cumulative CSV)
# ============================================================

def build_scorecard(cached_records, config, included_indices, top_k, eval_file_hash, dataset_hash, retrieval_only=False):
    accepted_statuses = (
        {"retrieval_complete", "answer_complete", "faithfulness_complete", "complete"}
        if retrieval_only else {"complete"}
    )
    completed = [
        r for index, r in cached_records.items()
        if index in included_indices and r.get("status") in accepted_statuses
    ]
    if not completed:
        raise ValueError("No completed evaluation records.")

    def collect(field):
        return [float(r[field]) for r in completed if r[field] is not None]

    recall_scores = collect("recall_at_5")
    mrr_scores = collect("mrr")
    ndcg_scores = collect("ndcg_at_5")
    faithfulness_scores = collect("faithfulness")
    answer_relevancy_scores = collect("answer_relevancy")
    unanswerable_count = sum(1 for r in completed if r.get("type") == "unanswerable")

    def avg(xs):
        return sum(xs) / len(xs) if xs else None

    return {
        "config_name": config.name(),
        "strategy": config.strategy,
        "chunk_size": config.chunk_size,
        "overlap": config.overlap,
        "similarity_threshold": config.similarity_threshold,
        "top_k": top_k,
        "eval_file_hash": eval_file_hash,
        "dataset_hash": dataset_hash,
        "num_questions": len(completed),
        "num_unanswerable": unanswerable_count,
        "num_scored_for_retrieval": len(recall_scores),
        "num_scored_for_faithfulness": len(faithfulness_scores),
        "num_scored_for_answer_relevancy": len(answer_relevancy_scores),
        "recall_at_5": avg(recall_scores),
        "mrr": avg(mrr_scores),
        "ndcg_at_5": avg(ndcg_scores),
        "faithfulness": avg(faithfulness_scores),
        "answer_relevancy": avg(answer_relevancy_scores),
    }


def write_scorecard(scorecard, scorecard_path):
    """Upsert an experiment row instead of appending duplicates on rerun."""
    scorecard_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if scorecard_path.exists():
        with scorecard_path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    key_fields = ("config_name", "num_questions", "top_k", "eval_file_hash", "dataset_hash")
    target_key = tuple(str(scorecard.get(field, "")) for field in key_fields)
    kept = [
        row for row in rows
        if tuple(str(row.get(field, "")) for field in key_fields) != target_key
    ]
    kept.append(scorecard)

    with scorecard_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SCORECARD_FIELDS)
        writer.writeheader()
        writer.writerows(kept)


def print_scorecard(scorecard):
    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else ("N/A" if v is None else str(v))

    print()
    print("=" * 80)
    print(f"SCORECARD -- config: {scorecard['config_name']}")
    print("=" * 80)
    print(f"Questions evaluated: {scorecard['num_questions']} ({scorecard['num_unanswerable']} unanswerable)")
    print()
    print(f"Retrieval scored on: {scorecard['num_scored_for_retrieval']} questions")
    print(f"Recall@5: {fmt(scorecard['recall_at_5'])}")
    print(f"MRR: {fmt(scorecard['mrr'])}")
    print(f"nDCG@5: {fmt(scorecard['ndcg_at_5'])}")
    print(f"Faithfulness scored on: {scorecard['num_scored_for_faithfulness']} questions")
    print(f"Faithfulness: {fmt(scorecard['faithfulness'])}")
    print()
    print(f"Answer Relevancy scored on: {scorecard['num_scored_for_answer_relevancy']} questions")
    print(f"Answer Relevancy: {fmt(scorecard['answer_relevancy'])}")
    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-path", default=str(EVAL_PATH))
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--strategy", default="fixed", choices=["fixed", "recursive", "semantic"])
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--similarity-threshold", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--skip-ragas", action="store_true", help="retrieval-only smoke test, no Ragas/answer calls")
    parser.add_argument("--scorecard-path", default=str(SCORECARD_PATH))
    args = parser.parse_args()

    if args.top_k != 5:
        parser.error("Phase 4 score fields are Recall@5/nDCG@5; use --top-k 5 for comparable runs.")

    config = ChunkConfig(
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        similarity_threshold=args.similarity_threshold,
    )
    config_name = config.name()
    cache_path = cache_path_for(config_name)
    predictions_path = predictions_path_for(config_name)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    scorecard_path = Path(args.scorecard_path)

    api_key = None if args.skip_ragas else get_api_key()

    eval_path = Path(args.eval_path)
    print(f"Loading eval set from: {eval_path}")
    eval_records = load_eval_set(eval_path, max_questions=args.max_questions)
    print(f"Loaded records: {len(eval_records)}")

    cached_records = load_latest_records(cache_path)
    print(f"Cached question records for config '{config_name}': {len(cached_records)}")

    print()
    print(f"Initializing RAG system (config: {config_name})...")
    rag_kwargs = {"config": config, "top_k": args.top_k}
    if args.dataset:
        rag_kwargs["dataset_path"] = args.dataset
    rag = RAGSystem(**rag_kwargs)
    rag.build_index()

    if not args.skip_ragas:
        print()
        print("Creating async Gemini Ragas evaluator...")
        ragas_llm = create_ragas_llm(api_key)
        print("Loading local embeddings for Answer Relevancy...")
        ragas_embeddings = create_ragas_embeddings()
        faithfulness_metric = Faithfulness(llm=ragas_llm)
        answer_relevancy_metric = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
        print("Ragas metrics ready.")

    total = len(eval_records)

    for position, eval_record in enumerate(eval_records, start=1):
        original_index = eval_record["original_index"]

        print()
        print("=" * 80)
        print(f"[{config_name}] Processing record {position}/{total}")
        print(f"Question: {eval_record['question']}")
        print("=" * 80)

        if original_index in cached_records:
            cache_record = cached_records[original_index]
            cache_matches = (
                cache_record.get("question") == eval_record["question"]
                and cache_record.get("gold_spans") == eval_record["gold_spans"]
                and cache_record.get("config_name") == config_name
            )
            if cache_matches:
                print(f"Loaded cached status: {cache_record.get('status')}")
            else:
                print("Cached record is stale for this question/evidence; rebuilding it.")
                cache_record = create_cache_record(eval_record, config_name)
                save_cache_record(cache_record, cache_path)
                cached_records[original_index] = cache_record
        else:
            cache_record = create_cache_record(eval_record, config_name)
            save_cache_record(cache_record, cache_path)
            cached_records[original_index] = cache_record
            print("Created new cache record.")

        target_status = "retrieval_complete" if args.skip_ragas else "complete"
        if cache_record.get("status") == target_status or cache_record.get("status") == "complete":
            print("\nCACHE HIT: Question already at or past target stage. Skipping.")
            continue

        try:
            cache_record = run_retrieval_stage(rag, cache_record, cache_path, args.top_k)
            cached_records[original_index] = cache_record

            if args.skip_ragas:
                rebuild_predictions_file(cached_records, predictions_path)
                continue

            cache_record = run_answer_stage(rag, cache_record, cache_path)
            cached_records[original_index] = cache_record

            if cache_record.get("faithfulness") is None:
                print(f"\nWaiting {SECONDS_BETWEEN_API_STEPS}s before Faithfulness...")
                await asyncio.sleep(SECONDS_BETWEEN_API_STEPS)

            cache_record = await run_faithfulness_stage(cache_record, cache_path, faithfulness_metric)
            cached_records[original_index] = cache_record

            if cache_record.get("answer_relevancy") is None:
                print(f"\nWaiting {SECONDS_BETWEEN_API_STEPS}s before Answer Relevancy...")
                await asyncio.sleep(SECONDS_BETWEEN_API_STEPS)

            cache_record = await run_answer_relevancy_stage(cache_record, cache_path, answer_relevancy_metric)
            cached_records[original_index] = cache_record

            rebuild_predictions_file(cached_records, predictions_path)

            print("\nQUESTION COMPLETE")
            print(f"Recall@5: {cache_record['recall_at_5']}  MRR: {cache_record['mrr']}  nDCG@5: {cache_record['ndcg_at_5']}")
            print(f"Faithfulness: {cache_record['faithfulness']}  Answer Relevancy: {cache_record['answer_relevancy']}")

        except KeyboardInterrupt:
            print("\nStopped by user. All completed stages are cached. Rerun to resume.")
            raise
        except Exception as e:
            print("\n" + "=" * 80)
            print("EVALUATION STAGE FAILED")
            print("=" * 80)
            print(f"Current cache status: {cache_record.get('status')}")
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception message:\n{e}")
            print("Completed stages are already cached. Rerun to resume from the missing stage.")
            raise

        if position < total:
            print(f"\nWaiting {SECONDS_BETWEEN_QUESTIONS}s before next question...")
            await asyncio.sleep(SECONDS_BETWEEN_QUESTIONS)

    rebuild_predictions_file(cached_records, predictions_path)

    included_indices = {record["original_index"] for record in eval_records}
    target_statuses = {"retrieval_complete", "answer_complete", "faithfulness_complete", "complete"}
    completed_count = sum(
        1 for index, record in cached_records.items()
        if index in included_indices and record.get("status") in target_statuses
    )
    if completed_count > 0:
        dataset_path = Path(args.dataset) if args.dataset else rag.dataset_path
        scorecard = build_scorecard(
            cached_records,
            config,
            included_indices=included_indices,
            top_k=args.top_k,
            eval_file_hash=file_hash(eval_path),
            dataset_hash=file_hash(dataset_path),
            retrieval_only=args.skip_ragas,
        )
        write_scorecard(scorecard, scorecard_path)
        print_scorecard(scorecard)

    print()
    print(f"Stage cache: {cache_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Scorecard (upserted across configs): {scorecard_path}")

    send_phone_notification(
    title="RAG evaluation complete",
    message=(
        f"Finished evaluating {scorecard['num_questions']} questions.\n"
        f"Recall@5: {scorecard['recall_at_5']:.4f}\n"
        f"MRR: {scorecard['mrr']:.4f}\n"
        f"nDCG@5: {scorecard['ndcg_at_5']:.4f}"
    ),
)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        send_phone_notification(
            title="RAG evaluation failed",
            message=f"{type(exc).__name__}: {exc}",
        )
        raise
