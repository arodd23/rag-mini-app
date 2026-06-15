from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama

# load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# load documents
with open("data.txt", "r") as f:
    docs = f.readlines()

docs = [d.strip() for d in docs if d.strip()]

# embed documents
doc_embeddings = model.encode(docs)

# create FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings).astype('float32'))

print("✅ Indexed documents")

# ask a question
print("✅ About to ask for input...")
query = input("Ask a question: ")
print("✅ Got input:", query)

# embed query
query_embedding = model.encode([query])

# search top 2 docs
D, I = index.search(np.array(query_embedding).astype('float32'), 2)

retrieved_docs = [docs[i] for i in I[0]]

print("\n🔎 Retrieved docs:")
for doc in retrieved_docs:
    print("-", doc)

# build prompt
context = "\n".join(retrieved_docs)

prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}

Answer:
"""

# call LLM
response = ollama.chat(
    model='llama3.1:8b',
    messages=[{"role": "user", "content": prompt}]
)

print("\n💬 Answer:")
print(response['message']['content'])