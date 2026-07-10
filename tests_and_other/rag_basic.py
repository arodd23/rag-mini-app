'''
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.genai as genai
import os



# load model
model_llm = genai.GenerativeModel("gemini-1.5-flash-latest")

# embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# load documents
with open("data.txt", "r") as f:
    docs = f.readlines()

docs = [d.strip() for d in docs if d.strip()]

# embed documents
doc_embeddings = embedding_model.encode(docs)

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
query_embedding = embedding_model.encode([query])

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

# call Gemini
response = model_llm.generate_content(prompt)

print("\n💬 Answer:")
print(response['message']['content'])
'''
'''
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from google import genai

# ✅ PUT YOUR API KEY HERE
client = genai.Client(api_key="YOUR_API_KEY_HERE")

# ✅ embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# ✅ load documents
with open("data.txt", "r") as f:
    docs = f.readlines()

docs = [d.strip() for d in docs if d.strip()]

# ✅ embed documents
doc_embeddings = embedding_model.encode(docs)

dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings).astype('float32'))

print("✅ Indexed documents")

# ✅ get user query
query = input("Ask a question: ")

# ✅ embed query
query_embedding = embedding_model.encode([query])

D, I = index.search(np.array(query_embedding).astype('float32'), 2)
retrieved_docs = [docs[i] for i in I[0]]

print("\n🔎 Retrieved docs:")
for doc in retrieved_docs:
    print("-", doc)

# ✅ build prompt
context = "\n".join(retrieved_docs)

prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}

Answer:
"""

# ✅ call Gemini (correct API + correct model)
response = client.models.generate_content(
    model="gemini-1.5-flash-latest",
    contents=prompt
)

print("\n💬 Answer:")
print(response.text)
'''