"""
run_chunking_sweep.py

Phase 4: sweeps chunk strategy x size x overlap, calling evaluate.py once
per config via subprocess (so one config failing doesn't kill the rest,
and each config's own per-question cache lets you resume a half-finished
sweep by just re-running this script).

Grid:
    fixed, recursive:  chunk_size in {256, 512, 1024} x overlap_pct in {0%, 10%, 20%}  = 18 runs
    semantic:          chunk_size (max chars) in {256, 512, 1024}, fixed similarity   =  3 runs
                        (overlap isn't a meaningful concept for semantic grouping)

That's 21 full evaluate.py runs. Time aside, that's still a large number of
LLM calls (21 configs x questions-per-config x ~3 calls/question for
answer+faithfulness+relevancy). Default here is a 40-question subset per
config so the sweep is about *comparing* configs -- run the full eval set
once, at the end, on whichever config wins.

Usage:
    python run_chunking_sweep.py
    python run_chunking_sweep.py --max-questions 170     # full eval set, every config
    python run_chunking_sweep.py --dry-run               # print commands only
"""

import argparse
import subprocess
import sys
import requests

CHUNK_SIZES = [256, 512, 1024]
OVERLAP_FRACTIONS = [0.0, 0.10, 0.20]
SEMANTIC_SIMILARITY_THRESHOLD = 0.6

DEFAULT_SWEEP_MAX_QUESTIONS = 40

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

def build_grid():
    grid = []
    for strategy in ("fixed", "recursive"):
        for size in CHUNK_SIZES:
            for frac in OVERLAP_FRACTIONS:
                grid.append({"strategy": strategy, "chunk_size": size, "overlap": int(size * frac)})
    for size in CHUNK_SIZES:
        grid.append({
            "strategy": "semantic",
            "chunk_size": size,
            "overlap": 0,
            "similarity_threshold": SEMANTIC_SIMILARITY_THRESHOLD,
        })
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-questions", type=int, default=DEFAULT_SWEEP_MAX_QUESTIONS)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--eval-path", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-id", type=int, default=0, help="zero-based worker/clone id")
    parser.add_argument("--num-workers", type=int, default=1, help="number of parallel clones")
    parser.add_argument("--scorecard-path", default=None, help="per-clone scorecard CSV path")
    parser.add_argument("--skip-ragas", action="store_true", help="retrieval-only sweep")
    args = parser.parse_args()

    if args.num_workers < 1:
        parser.error("--num-workers must be >= 1")
    if not 0 <= args.worker_id < args.num_workers:
        parser.error("--worker-id must satisfy 0 <= worker-id < num-workers")

    full_grid = build_grid()
    grid = [cfg for index, cfg in enumerate(full_grid) if index % args.num_workers == args.worker_id]
    print(
        f"Worker {args.worker_id}/{args.num_workers - 1} owns {len(grid)} of "
        f"{len(full_grid)} configs; max_questions={args.max_questions} each."
    )

    failures = []

    for i, cfg in enumerate(grid, start=1):
        cmd = [
            sys.executable, "evaluate.py",
            "--strategy", cfg["strategy"],
            "--chunk-size", str(cfg["chunk_size"]),
            "--overlap", str(cfg["overlap"]),
            "--top-k", str(args.top_k),
            "--max-questions", str(args.max_questions),
        ]
        if cfg["strategy"] == "semantic":
            cmd += ["--similarity-threshold", str(cfg["similarity_threshold"])]
        if args.eval_path:
            cmd += ["--eval-path", args.eval_path]
        if args.scorecard_path:
            cmd += ["--scorecard-path", args.scorecard_path]
        if args.skip_ragas:
            cmd += ["--skip-ragas"]

        print()
        print("=" * 80)
        print(f"[{i}/{len(grid)}] {' '.join(cmd)}")
        print("=" * 80)

        if args.dry_run:
            continue

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Config {cfg} exited with code {result.returncode}. Continuing to next config.")
            failures.append(cfg)

    print()
    print("Sweep done.")
    if failures:
        print(f"{len(failures)} config(s) did not finish cleanly: {failures}")
        print("Re-running this script will resume each from its cached progress.")
    print("See results/evaluation_scorecard.csv for the accumulated comparison table.")
    print("Run summarize_chunking_sweep.py to build the Phase 4 table + chart.")

    send_phone_notification(
    title="Phase 4 sweep complete",
    message="All assigned chunking configurations finished.",
)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        send_phone_notification(
            title="Phase 4 sweep failed",
            message=f"{type(exc).__name__}: {exc}",
        )
        raise
