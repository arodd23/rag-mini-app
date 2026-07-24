# Phase 4 code review

## Critical corrections made

1. **Span scoring no longer accepts any overlap as a full hit.** Recall uses union coverage of each gold span with a default 50% threshold.
2. **nDCG cannot double-count one gold span.** Each gold span receives relevance credit once, so nDCG stays in `[0, 1]`.
3. **Fixed chunk overlap follows the actual whitespace-adjusted boundary.** The former code advanced by a fixed step even when the end moved.
4. **Recursive overlap `0` no longer carries one sentence into the next chunk.**
5. **Short fragments are merged when possible instead of being silently deleted from the index.**
6. **Semantic centroid similarity is now cosine similarity.** The running centroid is normalized before the dot product.
7. **FAISS cache freshness includes both dataset and chunk-cache hashes.** Same chunk count is not enough to prove the index is current.
8. **`--skip-ragas` does not require `GEMINI_API_KEY`.**
9. **Scorecards are upserted, not blindly appended.** Rerunning an experiment replaces its row.
10. **Scorecards only summarize the eval records in the current invocation.** A 40-question run cannot accidentally include 170 cached questions.
11. **Parallel clones can shard the grid.** `run_chunking_sweep.py` now supports `--worker-id` and `--num-workers`.
12. **Parallel scorecards can be merged and deduplicated.** Use `merge_scorecards.py`.

## Recommended parallel workflow

Use one clean source repo, commit it, then create clones/worktrees only after the corpus and eval set are frozen and checksummed.

For three clones:

```bash
# clone 0
python run_chunking_sweep.py --worker-id 0 --num-workers 3 \
  --scorecard-path results/scorecard_worker0.csv

# clone 1
python run_chunking_sweep.py --worker-id 1 --num-workers 3 \
  --scorecard-path results/scorecard_worker1.csv

# clone 2
python run_chunking_sweep.py --worker-id 2 --num-workers 3 \
  --scorecard-path results/scorecard_worker2.csv
```

Then merge from a location that can see all three files:

```bash
python merge_scorecards.py \
  clone0/results/scorecard_worker0.csv \
  clone1/results/scorecard_worker1.csv \
  clone2/results/scorecard_worker2.csv \
  --output results/evaluation_scorecard.csv
```

### API warning

Parallel clones multiply API request rate. Use separate API keys/quotas, or increase per-process delays by roughly the worker count. The safest option is to parallelize chunk building/retrieval first with `--skip-ragas`, select promising configs, and run full Ragas scoring on fewer finalists.

## Original project files still needed before replacing your repo files

These generated files import or assume project code that was not in the zip. To do a true line-by-line integration check, provide:

- the original `rag_core.py`
- the original `evaluate.py`
- `eval_common.py`
- `requirements.txt` or `pyproject.toml`
- 2–3 sample lines from `data/eval/eval_set_v2_raw.jsonl`
- the actual location/schema of `gov_report_sample_2k.json`
- `.env.example` with variable names only, not keys

The attached version compiles and its local unit tests pass, but live Gemini/Ragas compatibility cannot be proven without those files and installed dependency versions.
