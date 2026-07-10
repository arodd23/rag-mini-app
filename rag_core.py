import hashlib
import json
import os
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer


class RAGSystem:
    def __init__(
        self,
        data_path="data/docs.txt",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=500,
        overlap=50,
        cache_dir="data/cache",
        generation_model="gemini-2.5-flash",
    ):
        load_dotenv()

        # ====================================================
        # CONFIG
        # ====================================================

        self.data_path = Path(data_path)

        self.embedding_model = embedding_model

        self.chunk_size = chunk_size

        self.overlap = overlap

        self.cache_dir = Path(cache_dir)

        self.generation_model = generation_model

        # ====================================================
        # CACHE PATHS
        # ====================================================

        self.chunks_cache_path = (
            self.cache_dir / "chunks.json"
        )

        self.embeddings_cache_path = (
            self.cache_dir / "chunk_embeddings.npy"
        )

        self.faiss_cache_path = (
            self.cache_dir / "faiss.index"
        )

        self.metadata_cache_path = (
            self.cache_dir / "cache_metadata.json"
        )

        # ====================================================
        # RUNTIME STATE
        # ====================================================

        self.chunks = []

        self.chunk_embeddings = None

        self.embedder = None

        self.index = None

        self.client = None


    # ========================================================
    # DOCUMENT LOADING
    # ========================================================

    def load_documents(self):
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Missing file: {self.data_path}"
            )

        if self.data_path.suffix.lower() == ".json":
            with self.data_path.open(
                "r",
                encoding="utf-8",
            ) as f:
                docs = json.load(f)

            return "\n\n".join(
                doc["text"]
                for doc in docs
            )

        return self.data_path.read_text(
            encoding="utf-8"
        )


    # ========================================================
    # CHUNKING
    # ========================================================

    def chunk_text(self, text):
        chunks = []

        start = 0

        step_size = (
            self.chunk_size
            - self.overlap
        )

        if step_size <= 0:
            raise ValueError(
                "chunk_size must be greater "
                "than overlap."
            )

        while start < len(text):
            end = start + self.chunk_size

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            start += step_size

        return chunks


    # ========================================================
    # HASHING
    # ========================================================

    def get_file_hash(self):
        """
        Hash the original corpus file.

        If docs.txt changes, the hash changes and
        the cache is automatically invalidated.
        """

        hasher = hashlib.sha256()

        with self.data_path.open("rb") as f:
            while True:
                data = f.read(1024 * 1024)

                if not data:
                    break

                hasher.update(data)

        return hasher.hexdigest()


    # ========================================================
    # CACHE METADATA
    # ========================================================

    def get_expected_metadata(self):
        return {
            "data_hash": self.get_file_hash(),
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
        }


    def load_cache_metadata(self):
        if not self.metadata_cache_path.exists():
            return None

        try:
            with self.metadata_cache_path.open(
                "r",
                encoding="utf-8",
            ) as f:
                return json.load(f)

        except (
            json.JSONDecodeError,
            OSError,
        ):
            return None


    def cache_is_valid(self):
        required_files = [
            self.chunks_cache_path,
            self.embeddings_cache_path,
            self.faiss_cache_path,
            self.metadata_cache_path,
        ]

        for path in required_files:
            if not path.exists():
                return False

        cached_metadata = (
            self.load_cache_metadata()
        )

        if cached_metadata is None:
            return False

        expected_metadata = (
            self.get_expected_metadata()
        )

        return (
            cached_metadata
            == expected_metadata
        )


    # ========================================================
    # CACHE SAVING
    # ========================================================

    def save_cache(self):
        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            "Saving chunks cache..."
        )

        with self.chunks_cache_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                self.chunks,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(
            "Saving embeddings cache..."
        )

        np.save(
            self.embeddings_cache_path,
            self.chunk_embeddings,
        )

        print(
            "Saving FAISS index cache..."
        )

        faiss.write_index(
            self.index,
            str(self.faiss_cache_path),
        )

        print(
            "Saving cache metadata..."
        )

        metadata = (
            self.get_expected_metadata()
        )

        with self.metadata_cache_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                metadata,
                f,
                indent=2,
            )


    # ========================================================
    # CACHE LOADING
    # ========================================================

    def load_cache(self):
        print(
            "Loading cached chunks..."
        )

        with self.chunks_cache_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            self.chunks = json.load(f)

        print(
            f"Total chunks: "
            f"{len(self.chunks)}"
        )

        print(
            "Loading cached embeddings..."
        )

        self.chunk_embeddings = np.load(
            self.embeddings_cache_path
        ).astype("float32")

        print(
            "Loading cached FAISS index..."
        )

        self.index = faiss.read_index(
            str(self.faiss_cache_path)
        )

        self.validate_loaded_cache()


    def validate_loaded_cache(self):
        """
        Basic corruption / mismatch checks.
        """

        num_chunks = len(self.chunks)

        num_embeddings = (
            self.chunk_embeddings.shape[0]
        )

        num_faiss_vectors = (
            self.index.ntotal
        )

        if num_chunks != num_embeddings:
            raise RuntimeError(
                "Cache mismatch: "
                f"{num_chunks} chunks but "
                f"{num_embeddings} embeddings."
            )

        if num_chunks != num_faiss_vectors:
            raise RuntimeError(
                "Cache mismatch: "
                f"{num_chunks} chunks but "
                f"{num_faiss_vectors} FAISS vectors."
            )


    # ========================================================
    # EMBEDDING MODEL
    # ========================================================

    def load_embedder(self):
        if self.embedder is not None:
            return

        print(
            f"Loading embedding model: "
            f"{self.embedding_model}"
        )

        self.embedder = SentenceTransformer(
            self.embedding_model
        )


    # ========================================================
    # INDEX BUILDING
    # ========================================================

    def build_index(
        self,
        force_rebuild=False,
    ):
        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.load_embedder()

        # ====================================================
        # USE CACHE
        # ====================================================

        if (
            not force_rebuild
            and self.cache_is_valid()
        ):
            print(
                "Valid RAG cache found."
            )

            try:
                self.load_cache()

                print(
                    "RAG system ready from cache."
                )

                return

            except Exception as e:
                print(
                    "Cache could not be loaded."
                )

                print(
                    f"Reason: {e}"
                )

                print(
                    "Rebuilding cache..."
                )

        # ====================================================
        # BUILD FROM SCRATCH
        # ====================================================

        print(
            "No valid RAG cache found."
        )

        print(
            "Loading documents..."
        )

        raw_text = self.load_documents()

        print(
            "Chunking documents..."
        )

        self.chunks = self.chunk_text(
            raw_text
        )

        print(
            f"Total chunks: "
            f"{len(self.chunks)}"
        )

        print(
            "Embedding chunks..."
        )

        self.chunk_embeddings = (
            self.embedder.encode(
                self.chunks,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            .astype("float32")
        )

        print(
            "Building FAISS index..."
        )

        dimension = (
            self.chunk_embeddings.shape[1]
        )

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(
            self.chunk_embeddings
        )

        self.save_cache()

        print(
            "RAG system ready."
        )


    # ========================================================
    # QUERY PREPARATION
    # ========================================================

    def prepare_search_query(
        self,
        query,
    ):
        if "bge" in self.embedding_model.lower():
            return (
                "Represent this sentence for "
                "searching relevant passages: "
                + query
            )

        return query


    # ========================================================
    # RETRIEVAL
    # ========================================================

    def retrieve(
        self,
        query,
        top_k=5,
    ):
        if (
            self.index is None
            or self.embedder is None
        ):
            raise RuntimeError(
                "Index not built. "
                "Call build_index() first."
            )

        search_query = (
            self.prepare_search_query(
                query
            )
        )

        query_embedding = (
            self.embedder.encode(
                [search_query],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            .astype("float32")
        )

        top_k = min(
            top_k,
            len(self.chunks),
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k,
        )

        results = []

        for score, idx in zip(
            scores[0],
            indices[0],
        ):
            if idx == -1:
                continue

            results.append({
                "chunk_id": int(idx),
                "score": float(score),
                "text": self.chunks[int(idx)],
            })

        return results


    # ========================================================
    # GEMINI CLIENT
    # ========================================================

    def get_generation_client(self):
        """
        Lazy-load Gemini.

        Retrieval-only workflows no longer require
        an API key.
        """

        if self.client is not None:
            return self.client

        api_key = os.getenv(
            "GEMINI_API_KEY"
        )

        if not api_key:
            raise ValueError(
                "Missing GEMINI_API_KEY. "
                "Add it to your .env file before "
                "generating answers."
            )

        os.environ.pop(
            "GOOGLE_API_KEY",
            None,
        )

        self.client = genai.Client(
            api_key=api_key
        )

        return self.client


    # ========================================================
    # PROMPT
    # ========================================================

    def build_answer_prompt(
        self,
        question,
        retrieved_chunks,
    ):
        context = "\n\n".join(
            (
                f"[Chunk {item['chunk_id']}]\n"
                f"{item['text']}"
            )
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


    # ========================================================
    # ANSWER FROM EXISTING CHUNKS
    # ========================================================

    def answer_from_chunks(
        self,
        question,
        retrieved_chunks,
    ):
        """
        Generate an answer from chunks that were already
        retrieved.

        This prevents evaluate.py from retrieving twice.
        """

        client = (
            self.get_generation_client()
        )

        prompt = self.build_answer_prompt(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )

        response = (
            client.models.generate_content(
                model=self.generation_model,
                contents=prompt,
            )
        )

        if not response.text:
            raise RuntimeError(
                "Gemini returned an empty response."
            )

        return response.text.strip()


    # ========================================================
    # FULL RAG QUESTION
    # ========================================================

    def answer_question(
        self,
        question,
        top_k=5,
    ):
        retrieved_chunks = self.retrieve(
            question,
            top_k=top_k,
        )

        answer = self.answer_from_chunks(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )

        return answer, retrieved_chunks