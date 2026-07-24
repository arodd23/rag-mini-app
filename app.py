import streamlit as st
from pathlib import Path

st.set_page_config(page_title="RAG Mini-App", layout="wide")

st.title("RAG Mini-App")
st.write("Page loaded successfully.")

st.info("If you can see this, Streamlit is working and app.py is rendering.")

DATA_PATH = Path("data/dataset/gov_report_sample+2k.jsonl")

if not DATA_PATH.exists():
    st.error("Missing data/docs.txt. Run python load_squad.py first.")
    st.stop()

question = st.text_input("Ask a question:")

run_button = st.button("Run RAG")

if not run_button:
    st.stop()

if not question.strip():
    st.warning("Type a question first.")
    st.stop()

st.write("Step 1: Starting imports...")

try:
    st.write("Import A: os")
    import os
    st.success("os imported")

    st.write("Import B: numpy")
    import numpy as np
    st.success("numpy imported")

    st.write("Import C: dotenv")
    from dotenv import load_dotenv
    st.success("dotenv imported")

    st.write("Import D: google genai")
    from google import genai
    st.success("google genai imported")

    st.write("Import E: faiss")
    import faiss
    st.success("faiss imported")
    
    st.write("Import F: SentenceTransformer")
    import torch
    st.success(f"torch imported: {torch.__version__}")

    import transformers
    st.success(f"transformers imported: {transformers.__version__}")

    from sentence_transformers import SentenceTransformer
    st.success("SentenceTransformer imported")

except Exception as e:
    st.error(f"Import F failed: {type(e).__name__}: {e}")
    st.stop()


st.success("Step 2: Imports complete.")

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

with st.spinner("Loading docs and building index..."):
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("Missing GEMINI_API_KEY in .env")
        st.stop()

    raw_text = DATA_PATH.read_text(encoding="utf-8")
    chunks = chunk_text(raw_text)

    st.write(f"Step 3: Created {len(chunks)} chunks.")

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    chunk_embeddings = embedder.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype("float32")

    st.write("Step 4: Embeddings created.")

    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(chunk_embeddings)

    st.write("Step 5: FAISS index built.")

    client = genai.Client(api_key=api_key)

st.success("RAG system ready.")

with st.spinner("Retrieving chunks and generating answer..."):
    query_embedding = embedder.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    top_k = min(5, len(chunks))
    scores, indices = index.search(query_embedding, top_k)

    retrieved = []
    for score, idx in zip(scores[0], indices[0]):
        retrieved.append({
            "chunk_id": int(idx),
            "score": float(score),
            "text": chunks[int(idx)]
        })

    context = "\n\n".join(
        f"[Chunk {item['chunk_id']}]\n{item['text']}"
        for item in retrieved
    )

    prompt = f"""
You are a helpful RAG assistant.

Answer the user's question using ONLY the context below.
If the answer is not in the context, say:
"I don't have enough information to answer that."

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

st.subheader("Answer")
st.write(response.text)

with st.expander("Retrieved Chunks"):
    for item in retrieved:
        st.markdown(f"### Chunk {item['chunk_id']} | Score: {item['score']:.4f}")
        st.write(item["text"])