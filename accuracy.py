import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load your dataset
df = pd.read_csv("medquad_clean.csv").dropna(subset=["Question","Answer"]).reset_index(drop=True)

# Same embedding model as your system
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Embed answers
answer_embs = model.encode(df["Answer"].tolist(), convert_to_numpy=True, batch_size=64)
faiss.normalize_L2(answer_embs)

# Build FAISS index
index = faiss.IndexFlatIP(answer_embs.shape[1])
index.add(answer_embs)

# Embed questions
question_embs = model.encode(df["Question"].tolist(), convert_to_numpy=True, batch_size=64)
faiss.normalize_L2(question_embs)

# Search Top-1
D, I = index.search(question_embs, 1)

correct = sum(I[i][0] == i for i in range(len(df)))
accuracy = correct / len(df)

print("Top-1 Retrieval Accuracy:", accuracy)
