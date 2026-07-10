from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# test embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
emb = model.encode(["hello world"])

print("Embedding shape:", emb.shape)

# test faiss
index = faiss.IndexFlatL2(384)
index.add(np.array(emb).astype('float32'))

D, I = index.search(np.array(emb).astype('float32'), 1)

print("FAISS works:", I)
