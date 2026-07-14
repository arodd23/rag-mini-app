"""Core dense-retrieval RAG system aligned with gen_eval.py chunk IDs."""

import hashlib
import json
import os
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

from corpus import load_raw_docs
from build_chunk_index import load_chunk_index, DEFAULT_DATASET_PATH


class RAGSystem:
    def __init__(
        self,
        data_path=DEFAULT_DATASET_PATH,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        cache_dir="data/cache",
        generation_model="gemini-2.5-flash",
    ):
        load_dotenv()

        self.data_path = Path(data_path)
        self.embedding_model = embedding_model
        self.cache_dir = Path(cache_dir)
        self.generation_model = generation_model

        self.chunks_cache_path = self.cache_dir / "chunks.json"
        self.embeddings_cache_path = self.cache_dir / "chunk_embeddings.npy"
        self.faiss_cache_path = self.cache_dir / "faiss.index"
        self.metadata_cache_path = self.cache_dir / "cache_metadata.json"

        self.chunks = []
        self.chunk_embeddings = None
        self.embedder = None
        self.index = None
        self.client = None

    def load_chunks(self):
        """
        Rebuild retrievable chunks from the same canonical chunk index used by
        gen_eval.py. This is the key invariant: eval gold_chunk_ids and FAISS
        result chunk_ids come from one shared source of truth.
        """
        chunk_records = load_chunk_index(dataset_path=self.data_path)
        docs = load_raw_docs(self.data_path)
        docs_by_id = {int(doc["doc_id"]): doc["text"] for doc in docs}

        chunks = []
        seen_ids = set()

        for record in chunk_records:
            chunk_id = int(record["chunk_id"])
            doc_id = int(record["doc_id"])
            start = int(record["start"])
            end = int(record["end"])

            if chunk_id in seen_ids:
                raise ValueError(f"Duplicate chunk_id in chunk index: {chunk_id}")
            if doc_id not in docs_by_id:
                raise ValueError(
                    f"Chunk {chunk_id} references missing doc_id {doc_id}."
                )

            text = docs_by_id[doc_id][start:end]
            if not text.strip():
                raise ValueError(f"Chunk {chunk_id} has empty text.")

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "start": start,
                    "end": end,
                    "text": text,
                }
            )
            seen_ids.add(chunk_id)

        # FAISS row positions do not have to equal chunk IDs, but stable index
        # ordering makes cache behavior and debugging deterministic.
        chunks.sort(key=lambda item: item["chunk_id"])
        return chunks

    def get_file_hash(self):
        if not self.data_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {self.data_path}")

        hasher = hashlib.sha256()
        with self.data_path.open("rb") as file_obj:
            for block in iter(lambda: file_obj.read(1024 * 1024), b""):
                hasher.update(block)
        return hasher.hexdigest()

    def get_chunk_index_hash(self):
        """Hash canonical chunk metadata so cache invalidates after re-chunking."""
        records = load_chunk_index(dataset_path=self.data_path)
        canonical = json.dumps(records, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def get_expected_metadata(self):
        return {
            "data_path": str(self.data_path.resolve()),
            "data_hash": self.get_file_hash(),
            "chunk_index_hash": self.get_chunk_index_hash(),
            "embedding_model": self.embedding_model,
        }

    def load_cache_metadata(self):
        if not self.metadata_cache_path.exists():
            return None
        try:
            with self.metadata_cache_path.open("r", encoding="utf-8") as file_obj:
                return json.load(file_obj)
        except (json.JSONDecodeError, OSError):
            return None

    def cache_is_valid(self):
        required = [
            self.chunks_cache_path,
            self.embeddings_cache_path,
            self.faiss_cache_path,
            self.metadata_cache_path,
        ]
        return all(path.exists() for path in required) and (
            self.load_cache_metadata() == self.get_expected_metadata()
        )

    def save_cache(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with self.chunks_cache_path.open("w", encoding="utf-8") as file_obj:
            json.dump(self.chunks, file_obj, ensure_ascii=False, indent=2)
        np.save(self.embeddings_cache_path, self.chunk_embeddings)
        faiss.write_index(self.index, str(self.faiss_cache_path))
        with self.metadata_cache_path.open("w", encoding="utf-8") as file_obj:
            json.dump(self.get_expected_metadata(), file_obj, indent=2)

    def load_cache(self):
        with self.chunks_cache_path.open("r", encoding="utf-8") as file_obj:
            self.chunks = json.load(file_obj)
        self.chunk_embeddings = np.load(self.embeddings_cache_path).astype("float32")
        self.index = faiss.read_index(str(self.faiss_cache_path))
        self.validate_loaded_cache()

    def validate_loaded_cache(self):
        counts = (
            len(self.chunks),
            self.chunk_embeddings.shape[0],
            self.index.ntotal,
        )
        if len(set(counts)) != 1:
            raise RuntimeError(
                "Cache mismatch: "
                f"chunks={counts[0]}, embeddings={counts[1]}, faiss={counts[2]}"
            )

        chunk_ids = [int(chunk["chunk_id"]) for chunk in self.chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise RuntimeError("Cached chunks contain duplicate chunk IDs.")

    def load_embedder(self):
        if self.embedder is None:
            print(f"Loading embedding model: {self.embedding_model}")
            self.embedder = SentenceTransformer(self.embedding_model)

    def build_index(self, force_rebuild=False):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.load_embedder()

        if not force_rebuild and self.cache_is_valid():
            try:
                self.load_cache()
                print(f"RAG system ready from cache. Total chunks: {len(self.chunks)}")
                return
            except Exception as exc:
                print(f"Cache load failed ({exc}); rebuilding.")

        self.chunks = self.load_chunks()
        print(f"Loaded canonical chunks: {len(self.chunks)}")

        self.chunk_embeddings = self.embedder.encode(
            [chunk["text"] for chunk in self.chunks],
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

    def retrieve(self, query, top_k=5):
        if self.index is None or self.embedder is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        query_embedding = self.embedder.encode(
            [self.prepare_search_query(query)],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        top_k = min(max(int(top_k), 1), len(self.chunks))
        scores, row_indices = self.index.search(query_embedding, top_k)

        results = []
        for score, row_index in zip(scores[0], row_indices[0]):
            if row_index < 0:
                continue
            chunk = self.chunks[int(row_index)]
            results.append(
                {
                    "chunk_id": int(chunk["chunk_id"]),
                    "doc_id": int(chunk["doc_id"]),
                    "start": int(chunk["start"]),
                    "end": int(chunk["end"]),
                    "score": float(score),
                    "text": chunk["text"],
                }
            )
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
        context = "\n\n".join(
            f"[Chunk {item['chunk_id']}]\n{item['text']}"
            for item in retrieved_chunks
        )
        return f"""
You are a helpful RAG assistant.

Answer the user's question using ONLY the context below.

If the answer is not in the context, say exactly:
"I don't have enough information to answer that."

Context:
{context}

Question:
{question}

Answer:
""".strip()

    def answer_from_chunks(self, question, retrieved_chunks):
        client = self.get_generation_client()
        response = client.models.generate_content(
            model=self.generation_model,
            contents=self.build_answer_prompt(question, retrieved_chunks),
        )
        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")
        return response.text.strip()

    def answer_question(self, question, top_k=5):
        retrieved_chunks = self.retrieve(question, top_k=top_k)
        answer = self.answer_from_chunks(question, retrieved_chunks)
        return answer, retrieved_chunks