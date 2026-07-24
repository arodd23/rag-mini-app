# Phase 4 integration notes

This package has been reconciled with Aditi's original `rag_core.py`,
`evaluate.py`, and `eval_common.py`.

## Confirmed project paths

- Corpus: `data/dataset/gov_report_sample_2k.json`
- Eval set: `data/eval/eval_set_v2_raw.jsonl`

## Important behavior

- The original Phase 3 `RAGSystem` interface remains supported.
- Phase 4 adds `ChunkConfig` and config-specific chunk/FAISS/evaluation caches.
- Retrieval metrics use durable `gold_spans`, not stale `gold_chunk_ids`.
- `--skip-ragas` performs retrieval-only sweeps and still writes scorecard rows.
- Ragas remains the evaluator for full finalist runs.
- No `requirements.txt` or `pyproject.toml` is required to use these replacement files.

## Recommended order

1. Back up or commit the current repository.
2. Copy these files into the repository root.
3. Run `python -m pytest -q`.
4. Run a one-question retrieval smoke test:
   `python evaluate.py --max-questions 1 --skip-ragas`
5. Run a sharded retrieval sweep in each clone/worktree.
6. Merge scorecards, summarize, and select finalists.
7. Run full Ragas evaluation only on the finalists.
