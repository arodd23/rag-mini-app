# Project: Build and Probe a RAG Mini-App

---

## How to use this plan

- **Build from primitives.** Use `sentence-transformers` + FAISS + `rank_bm25` directly
rather than a big framework. You will understand every moving part. Treat LlamaIndex and
LangChain as reference implementations to *read*, not to hide behind.
- **Measure everything.** From Phase 3 on, no change is "better" until the eval harness says
so. Eyeballing a demo is not evidence.
- **One git repo, committed daily.** A short `NOTES.md` per phase: what you tried, the
numbers, what you concluded.
- **Ask early.** If you are stuck for more than ~2 hours, post in the daily standup.

## Core stack (all free)

- Python 3.11, `numpy`, `pandas`, `matplotlib`
- `sentence-transformers` (embeddings + rerankers), `faiss-cpu` (vector index),
`rank_bm25` (lexical), `scikit-learn`
- A multilingual embedding model for later phases: `intfloat/multilingual-e5-small` or
`BAAI/bge-m3`
- An LLM: local via [Ollama](https://ollama.com) (e.g. `llama3.1:8b`) or a hosted API
- UI: [Streamlit](https://docs.streamlit.io); Eval: [Ragas](https://docs.ragas.io/en/stable/)

## Foundational references (bookmark these)

- Hugging Face Cookbook, [Advanced RAG](https://huggingface.co/learn/cookbook/advanced_rag)
- [NirDiamant/RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques): a notebook per technique you will implement. Your single best reference.
- Sentence Transformers: [Quickstart](https://www.sbert.net/docs/quickstart.html), [Semantic Search](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html), [Retrieve & Re-Rank](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html), [Pretrained models](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html)
- Pinecone Learn (vendor-neutral enough): [learn center](https://www.pinecone.io/learn/)

## Corpus

Pick one bounded, public set (~500 to 2,000 documents so iteration is fast on a laptop):

- A themed Wikipedia subset via `[wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia)`, or
- [SQuAD](https://huggingface.co/datasets/rajpurkar/squad) / [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa) context passages (they ship gold questions, handy to sanity-check your own synthetic questions). [SQuAD 2.0](https://huggingface.co/datasets/rajpurkar/squad_v2) additionally has unanswerable questions, useful in Phase 8.

Recommendation: a themed Wikipedia subset for the app, with a few hundred SQuAD/HotpotQA
items kept aside as a labeled reference set.

---

## Recurring thread: Synthetic Data Generation

You build every eval set yourself by generating data with an LLM, then validating it. This
is a real, transferable skill, assessed across the project:

- **Phase 2:** grounded question/answer pairs from corpus chunks.
- **Phase 3:** harder cases (paraphrases, multi-hop, unanswerable / "should abstain").
- **Phase 9:** translated queries for cross-lingual evaluation.
- **Phase 10:** evolving user-memory timelines and deliberately conflicting facts.
Each time: generate, validate against a schema, **human-review a sample**, and measure how
good the synthetic data actually is. Lesson: synthetic data is only as good as your review.
References: Ragas [test set generation](https://docs.ragas.io/en/stable/getstarted/rag_testset_generation/), [distilabel](https://distilabel.argilla.io/latest/).

---

# Phase 1: Concepts + a naive RAG pipeline end to end

**Goal:** understand retrieve-then-read and stand up the simplest version that works.

**Learn**

- Embeddings and similarity search: Sentence Transformers [Quickstart](https://www.sbert.net/docs/quickstart.html) and [Semantic Search](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html).
- The RAG pattern end to end: a "RAG from scratch" walkthrough in [RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques).
- FAISS basics: [FAISS wiki: Getting started](https://github.com/facebookresearch/faiss/wiki/Getting-started).

**Tasks**

1. Set up the repo, virtualenv, install the core stack. Get Ollama (or an API) working from
  Python.
2. Load the corpus into `{doc_id, text}` records.
3. Naive fixed-size chunking (e.g. 500 chars, 50 overlap).
4. Embed chunks with `sentence-transformers/all-MiniLM-L6-v2`; store in a FAISS
  `IndexFlatIP` (L2-normalized vectors so inner product = cosine).
5. Query path: embed query, retrieve top-5, format into a prompt, call the LLM, print the
  answer and the chunks used.

**Build:** a CLI `ask.py "your question"`.

**Done when:** 5 questions return answers grounded in retrieved chunks, and `NOTES.md`
explains each stage in your own words.

**Stretch:** swap in `BAAI/bge-small-en-v1.5` and eyeball the difference.

---

# Phase 2: Wrap it in an app + your first synthetic eval set

**Goal:** make it usable, and create data to measure against.

**Learn**

- Streamlit: [Get started](https://docs.streamlit.io/get-started).
- LLM-based data generation: Ragas [test set generation](https://docs.ragas.io/en/stable/getstarted/rag_testset_generation/).

**Tasks**

1. Streamlit app: query box, answer, expandable "retrieved chunks" panel. Log every query
  and its retrieved chunk ids.
2. `gen_eval.py`: for ~80 sampled chunks, prompt the LLM to write one question whose answer
  is fully in that chunk. Store `{question, answer, gold_chunk_id}` as JSONL; validate the
   shape on write.
3. Review ~20 generated items; drop the bad ones; record your reject rate.

**Build:** the app + `eval_set_v0.jsonl` + the generation script.

**Done when:** app runs and you have ~60 reviewed Q/A items with a noted reject rate.

---

# Phase 3: The evaluation harness (the backbone of the whole project)

**Goal:** measure retrieval and answer quality automatically. Everything later depends on it.

**Learn**

- Retrieval metrics: recall@k, MRR, [nDCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain).
- Answer metrics and LLM-as-judge: Ragas [metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/) (`context_recall`, `context_precision`, `faithfulness`, `answer_relevancy`) and the [simple RAG eval guide](https://docs.ragas.io/en/stable/getstarted/rag_eval/).

**Tasks**

1. Implement `recall_at_k`, `mrr`, `ndcg_at_k` yourself in `metrics.py`; unit-test on a tiny
  hand-made example.
2. `evaluate.py`: given the eval set and the system, compute retrieval recall@k and MRR, then
  Ragas faithfulness and answer-relevancy on the answers. Output a one-row scorecard (CSV).
3. Expand the eval set to ~200 items including paraphrased, multi-hop, and ~40 unanswerable
  questions (the right behavior is to abstain; you will use these in Phase 8). Re-review a
   sample.

**Build:** `evaluate.py` producing the naive-system baseline scorecard.

**Done when:** one command prints recall@5, MRR, faithfulness, and answer-relevancy. This is
the baseline every later change is compared against.

---

# Phase 4: Track C, chunking part 1 (strategy and size)

**Goal:** measure how much chunking alone changes quality.

**Learn**

- Chunking strategies: Greg Kamradt's ["5 Levels of Text Splitting"](https://github.com/FullStackRetrieval-com/RetrievalTutorials) and the chunking section of the HF [Advanced RAG cookbook](https://huggingface.co/learn/cookbook/advanced_rag).
- Token-aware vs character-aware splitting.

**Tasks**

1. Three chunkers behind one interface: fixed-size (baseline), recursive (paragraph then
  sentence), semantic (group adjacent sentences by embedding similarity).
2. Sweep chunk size {256, 512, 1024} and overlap {0, 10%, 20%}.
3. Re-index and run `evaluate.py` per config; table + bar chart.

**Build:** results table + chart of chunking strategy and size vs recall@k / faithfulness.

**Done when:** you can state, with numbers, which config beats the Phase 3 baseline and by
how much.

---

# Phase 5: Track C, chunking part 2 (the techniques the team has not tried)

**Goal:** test parent-child retrieval and contextual retrieval.

**Learn**

- Parent-child / small-to-big and sentence-window: LlamaIndex [Auto-Merging Retriever](https://docs.llamaindex.ai/en/stable/examples/retrievers/auto_merging_retriever/) (read for the idea, implement a simple version).
- Contextual Retrieval: Anthropic's [Introducing Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) and the DataCamp [walkthrough](https://www.datacamp.com/tutorial/contextual-retrieval-anthropic).

**Tasks**

1. Parent-child retrieval: embed small child chunks, return the larger parent (or merge
  children) to the LLM. Compare to Phase 4's best.
2. Contextual retrieval: prepend a one-sentence LLM-generated description of where each chunk
  sits in its document before embedding. Re-index and eval. Note the indexing cost; mention
   prompt caching as the production fix.
3. Pick a recommended default chunking + retrieval-unit config from the data.

**Build:** `reports/chunking.md` with numbers and a recommendation.

**Done when:** you have a measured recommendation and can explain why contextual retrieval
helps (or does not) here.

---

# Phase 6: Track B, hybrid retrieval + query transforms

**Goal:** improve retrieval independent of chunking.

**Learn**

- Lexical vs dense; BM25: `[rank_bm25](https://github.com/dorianbrown/rank_bm25)`.
- Reciprocal Rank Fusion (RRF): the fusion notebook in [RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques).
- Query transforms: HyDE ([paper](https://arxiv.org/abs/2212.10496) + the HyDE notebook in RAG_Techniques) and multi-query.

**Tasks**

1. Add BM25 over the same chunks; implement RRF to fuse BM25 + dense; eval hybrid vs each.
2. Add query rewriting (LLM cleans the question into a search query); eval.
3. Add HyDE (LLM writes a hypothetical answer, embed that) and multi-query (3 to 4
  paraphrases, fuse with RRF); eval each.

**Build:** a results table ranking each technique by measured impact and rough cost.

**Done when:** you can rank hybrid, query-rewrite, HyDE, and multi-query by impact on this
corpus.

---

# Phase 7: Track B, reranking + consolidate "v2"

**Goal:** add reranking and lock in the best overall configuration.

**Learn**

- Cross-encoder reranking and the bi- vs cross-encoder tradeoff: Sentence Transformers
[Cross-Encoder usage](https://www.sbert.net/docs/cross_encoder/usage/usage.html) and [Retrieve & Re-Rank](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html).
- Reranker models: `cross-encoder/ms-marco-MiniLM-L6-v2` (fast) and
`[BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)` (stronger, multilingual, useful again in Phase 9).

**Tasks**

1. Retrieve top-50 with the best Phase-6 retriever, rerank with a cross-encoder, keep top-5.
  Eval with and without; note latency.
2. Assemble "v2": best chunking + best retrieval + reranking. Full eval vs the v0 baseline.
3. Write the retrieval bake-off report.

**Build:** `reports/retrieval.md` + a locked `v2` config with a clear before/after.

**Done when:** v2 beats v0 and you can attribute gains to specific techniques. Keep the
reranker's top score available; Phase 8 uses it as a confidence signal.

---

# Phase 8: Abstention ("I don't know")

**Goal:** make the system decline to answer when retrieval confidence is low, and measure it
as a precision/over-refusal tradeoff.

**Learn**

- Why abstention matters: an answer with no supporting context is a hallucination. Connect
this to the `faithfulness` metric from Phase 3.
- Confidence signals: top reranker score, the score gap between rank 1 and rank 2, and max
cosine similarity. Source of unanswerable examples: your Phase 3 set plus [SQuAD 2.0](https://huggingface.co/datasets/rajpurkar/squad_v2).

**Tasks**

1. For every eval query, log three candidate confidence signals (top reranker score,
  top1-minus-top2 gap, max cosine).
2. Grow the unanswerable set to ~60 (questions whose answer is not in the corpus). You now
  have answerable and unanswerable buckets.
3. Sweep a threshold on the chosen signal. Plot two curves against the threshold: fraction of
  answerable questions still answered, and fraction of unanswerable questions correctly
   abstained. This is an over-refusal vs miss tradeoff. Pick a threshold from the curve.
4. Wire abstention into the app: below threshold, reply "I do not have enough information to
  answer that" instead of generating.
5. Compare three abstention strategies and report which wins: (a) retrieval-threshold only,
  (b) prompt-guard only (instruct the LLM to refuse if the context lacks the answer), and
   (c) both together.

**Metrics**

- Abstention recall: of unanswerable questions, percent correctly abstained.
- Over-refusal rate: of answerable questions, percent wrongly abstained.
- Faithfulness on the full set before vs after abstention (should improve).

**Build:** abstention logic in the app + `reports/abstention.md` with the threshold tradeoff
curve and the chosen operating point.

**Done when:** the system abstains on most unanswerable questions while still answering most
answerable ones, with the threshold justified by the curve, and the over-refusal rate is
reported explicitly.

---

# Phase 9: Cross-lingual retrieval

**Goal:** answer queries asked in another language against the (English) corpus, and quantify
the quality gap versus monolingual.

**Learn**

- Multilingual embedding spaces align languages, so a non-English query can match English
text. Models: `[intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)` and `[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)`.
- Two approaches: native cross-lingual embedding vs translate-the-query-first.

**Tasks**

1. Swap the embedder to a multilingual model and re-baseline monolingual quality (it may
  differ from MiniLM; record it).
2. Build a cross-lingual eval subset: translate ~60 English eval questions into 3 EU
  languages (German, French, Spanish) with the LLM, keeping the same gold chunk ids (the
   docs stay English). Human-review a sample of translations (this is another synthetic-data
   exercise).
3. Run retrieval with non-English queries against the English index; measure recall@k per
  language and report the gap versus the English-query baseline.
4. Compare native multilingual embedding against translate-then-retrieve (translate the
  query to English first, then use the English pipeline). Report which wins per language.

**Build:** `reports/crosslingual.md` with a per-language results table and a recommendation.

**Done when:** you can quantify the cross-lingual recall gap per language and say whether
native multilingual embeddings or translate-then-retrieve works better on this corpus.

---

# Phase 10: Track F, memory-blended RAG (add a memory store)

**Goal:** blend document knowledge with an evolving user/session memory.

**Learn**

- Document knowledge vs user memory, and simple memory designs (read the relevant notebook in
RAG_Techniques).
- Recency and metadata filtering in retrieval.

**Tasks**

1. A memory store: a separate small index of `{fact, timestamp, source}` items (e.g. "user
  prefers metric units", "user's project is named X").
2. On a query, retrieve from both the document index and the memory index, merge (concatenate,
  dedupe, cap total context), and add a recency weight so newer memories rank higher.
3. **Synthetic data:** generate evolving user-memory timelines and ~80 queries that need a
  user fact + a document fact together, including cases where memory and documents disagree.

**Build:** the app answering questions that need a user fact + a document fact together.

**Done when:** there is at least one clear, demoable example where blending memory changes the
answer correctly.

---

# Phase 11: Track F, conflicts, dedup, and memory evaluation

**Goal:** handle the messy parts of memory and measure what it adds.

**Learn**

- Semantic near-duplicate detection (cosine threshold) and simple conflict resolution
(recency wins, or explicit override).

**Tasks**

1. Dedup on write: before adding a memory, check cosine similarity to existing memories;
  merge or skip near-duplicates above a tuned threshold.
2. Conflict handling: when two memories contradict, resolve by recency or an explicit "user
  corrected this" flag. Show the rule firing on a constructed example.
3. Build a memory-specific eval set and measure: does blending help memory-dependent
  questions without hurting pure-document questions?

**Build:** memory-blended RAG with dedup + conflict handling + `reports/memory.md`.

**Done when:** numbers show memory helps memory-dependent questions and does not regress
document questions, and dedup/conflict rules demonstrably work.

---

# Phase 12: Polish, final evaluation, write-up, demo

**Goal:** ship the artifact and communicate the findings.

**Tasks**

1. Make every strategy toggleable by config (chunking, retrieval, reranking, abstention,
  multilingual, memory on/off) so anyone can reproduce a configuration.
2. Final end-to-end eval: v0 vs v2 vs v2+memory, plus the abstention and cross-lingual
  results. One clean table and chart.
3. Write the README (how to run), a 3-page findings report, and a ~12-slide deck.
4. Deliver a ~20-minute demo ending with concrete "things worth considering" for the team.

**Build:** final repo + findings report + slide deck + demo.

**Done when:** a teammate can clone the repo, run one command, and reproduce your headline
numbers; and you have delivered the demo.

---

## Definition of done (whole project)

- A runnable, config-toggleable RAG app over a public corpus.
- An eval harness and a documented synthetic eval set you built and reviewed yourself.
- Five findings reports: chunking, retrieval, abstention, cross-lingual, memory, each backed
by numbers.
- A before/after scorecard and a team demo.

## Stretch goals

- Agentic / iterative retrieval (retrieve, judge, re-query) in the style of Corrective RAG.
- A fully on-device variant: small local embedder + small local LLM, with a quality vs
latency vs RAM comparison against the cloud setup.
- Fine-tune the reranker on your own synthetic pairs and measure the lift.

## What you are looking for / reporting

- Report numbers, not impressions. "Recall@5 went from 0.62 to 0.74" beats "it feels better."
- You can explain *why* a technique helped or did not on this corpus.
- Your synthetic data has a known, low reject rate because you review it.
- You ship something runnable at every phase boundary.

