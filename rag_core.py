"""Core dense-retrieval RAG system with per-chunk-config caches."""

import hashlib
import json
import os
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

from corpus import DEFAULT_CONFIG, file_hash, load_raw_docs
from build_chunk_index import DEFAULT_DATASET_PATH, cache_paths, load_chunk_index

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GENERATION_MODEL = "gemini-2.5-flash"


def _safe_model_name(model_name):
    return hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:10]


class RAGSystem:
    def __init__(
        self,
        data_path=DEFAULT_DATASET_PATH,
        *,
        dataset_path=None,
        config=None,
        top_k=5,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        cache_dir="data/cache",
        generation_model=DEFAULT_GENERATION_MODEL,
        gemini_client=None,
    ):
        load_dotenv()
        if dataset_path is not None:
            if data_path != DEFAULT_DATASET_PATH and Path(data_path) != Path(dataset_path):
                raise ValueError("Pass only one of data_path or dataset_path, not conflicting values.")
            data_path = dataset_path

        self.data_path = Path(data_path)
        self.dataset_path = self.data_path
        self.config = config or DEFAULT_CONFIG
        self.top_k = int(top_k)
        self.embedding_model = embedding_model
        self.cache_dir = Path(cache_dir)
        self.generation_model = generation_model

        model_tag = _safe_model_name(self.embedding_model)
        self.config_cache_dir = self.cache_dir / "rag_indexes" / self.config.name() / model_tag
        self.chunks_cache_path = self.config_cache_dir / "chunks.json"
        self.embeddings_cache_path = self.config_cache_dir / "chunk_embeddings.npy"
        self.faiss_cache_path = self.config_cache_dir / "faiss.index"
        self.metadata_cache_path = self.config_cache_dir / "cache_metadata.json"

        self.chunks = []
        self.chunk_embeddings = None
        self.embedder = None
        self.index = None
        self.client = gemini_client

    def load_embedder(self):
        if self.embedder is None:
            print(f"Loading embedding model: {self.embedding_model}")
            self.embedder = SentenceTransformer(self.embedding_model)
        return self.embedder

    def load_chunks(self):
        semantic_embedder = self.load_embedder() if self.config.strategy == "semantic" else None
        records = load_chunk_index(self.data_path, self.config, semantic_embedder)
        docs_by_id = {int(d["doc_id"]): d["text"] for d in load_raw_docs(self.data_path)}

        chunks = []
        seen_ids = set()
        for record in records:
            chunk_id = int(record["chunk_id"])
            doc_id = int(record["doc_id"])
            start = int(record["start"])
            end = int(record["end"])
            if chunk_id in seen_ids:
                raise ValueError(f"Duplicate chunk_id in chunk index: {chunk_id}")
            if doc_id not in docs_by_id:
                raise ValueError(f"Chunk {chunk_id} references missing doc_id {doc_id}.")
            if not (0 <= start < end <= len(docs_by_id[doc_id])):
                raise ValueError(f"Invalid offsets for chunk {chunk_id}: {start}:{end}")
            text = docs_by_id[doc_id][start:end]
            if not text.strip():
                raise ValueError(f"Chunk {chunk_id} has empty text.")
            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "start": start,
                "end": end,
                "text": text,
            })
            seen_ids.add(chunk_id)
        chunks.sort(key=lambda item: item["chunk_id"])
        return chunks

    def get_file_hash(self):
        if not self.data_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {self.data_path}")
        return file_hash(self.data_path)

    def get_chunk_index_hash(self):
        chunk_path, _ = cache_paths(self.config)
        if not chunk_path.exists():
            semantic_embedder = self.load_embedder() if self.config.strategy == "semantic" else None
            load_chunk_index(self.data_path, self.config, semantic_embedder)
        return file_hash(chunk_path)

    def get_expected_metadata(self):
        return {
            "data_path": str(self.data_path.resolve()),
            "data_hash": self.get_file_hash(),
            "chunk_index_hash": self.get_chunk_index_hash(),
            "chunk_config": self.config.as_dict(),
            "embedding_model": self.embedding_model,
        }

    def load_cache_metadata(self):
        if not self.metadata_cache_path.exists():
            return None
        try:
            return json.loads(self.metadata_cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def cache_is_valid(self):
        required = [self.chunks_cache_path, self.embeddings_cache_path,
                    self.faiss_cache_path, self.metadata_cache_path]
        return all(p.exists() for p in required) and self.load_cache_metadata() == self.get_expected_metadata()

    def save_cache(self):
        self.config_cache_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_cache_path.write_text(json.dumps(self.chunks, ensure_ascii=False), encoding="utf-8")
        np.save(self.embeddings_cache_path, self.chunk_embeddings)
        faiss.write_index(self.index, str(self.faiss_cache_path))
        self.metadata_cache_path.write_text(json.dumps(self.get_expected_metadata(), indent=2), encoding="utf-8")

    def load_cache(self):
        self.chunks = json.loads(self.chunks_cache_path.read_text(encoding="utf-8"))
        self.chunk_embeddings = np.load(self.embeddings_cache_path).astype("float32")
        self.index = faiss.read_index(str(self.faiss_cache_path))
        counts = (len(self.chunks), self.chunk_embeddings.shape[0], self.index.ntotal)
        if len(set(counts)) != 1:
            raise RuntimeError(f"Cache mismatch: chunks={counts[0]}, embeddings={counts[1]}, faiss={counts[2]}")

    def build_index(self, force_rebuild=False, force=None):
        if force is not None:
            force_rebuild = bool(force)
        self.load_embedder()
        if not force_rebuild and self.cache_is_valid():
            try:
                self.load_cache()
                print(f"RAG system ready from cache ({self.config.name()}). Total chunks: {len(self.chunks)}")
                return
            except Exception as exc:
                print(f"Cache load failed ({exc}); rebuilding.")

        self.chunks = self.load_chunks()
        print(f"Loaded chunks for {self.config.name()}: {len(self.chunks)}")
        self.chunk_embeddings = self.embedder.encode(
            [c["text"] for c in self.chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")
        self.index = faiss.IndexFlatIP(self.chunk_embeddings.shape[1])
        self.index.add(self.chunk_embeddings)
        self.save_cache()
        print("RAG system ready.")

    def prepare_search_query(self, query):
        if "bge" in self.embedding_model.lower():
            return "Represent this sentence for searching relevant passages: " + query
        return query

    def retrieve(self, query, top_k=None):
        if self.index is None or self.embedder is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")
        top_k = self.top_k if top_k is None else int(top_k)
        top_k = min(max(top_k, 1), len(self.chunks))
        q = self.embedder.encode([self.prepare_search_query(query)], convert_to_numpy=True,
                                 normalize_embeddings=True).astype("float32")
        scores, rows = self.index.search(q, top_k)
        results = []
        for score, row in zip(scores[0], rows[0]):
            if row < 0:
                continue
            c = self.chunks[int(row)]
            results.append({**c, "score": float(score)})
        return results

    def get_generation_client(self):
        if self.client is not None:
            return self.client
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY in .env file.")
        os.environ.pop("GOOGLE_API_KEY", None)
        self.client = genai.Client(api_key=api_key)
        return self.client

    def build_answer_prompt(self, question, retrieved_chunks):
        context = "\n\n".join(f"[Chunk {x['chunk_id']}]\n{x['text']}" for x in retrieved_chunks)
        return f'''You are a helpful RAG assistant.

Answer the user's question using ONLY the context below.

If the answer is not in the context, say exactly:
"I don't have enough information to answer that."

Context:
{context}

Question:
{question}

Answer:'''.strip()

    def answer_from_chunks(self, question, retrieved_chunks):
        response = self.get_generation_client().models.generate_content(
            model=self.generation_model,
            contents=self.build_answer_prompt(question, retrieved_chunks),
        )
        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")
        return response.text.strip()

    def answer_question(self, question, top_k=None):
        chunks = self.retrieve(question, top_k=top_k)
        return self.answer_from_chunks(question, chunks), chunks

    def answer(self, question, top_k=None):
        return self.answer_question(question, top_k=top_k)
