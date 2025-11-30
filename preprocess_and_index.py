# preprocess_and_index.py
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json
from typing import List

CSV_PATH = "medquad_clean.csv"  # your uploaded file
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # small & fast
INDEX_DIR = "index_data"
EMB_DIM = 384  # for all-MiniLM-L6-v2

os.makedirs(INDEX_DIR, exist_ok=True)

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Expecting columns: Question, Answer
    df = df.dropna(subset=["Question","Answer"]).reset_index(drop=True)
    return df

def build_embeddings(texts: List[str], model_name=EMB_MODEL_NAME):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, batch_size=64)
    return embeddings

def build_faiss_index(embeddings: np.ndarray, dim:int=EMB_DIM):
    index = faiss.IndexFlatIP(dim)  # inner product (use cosine if normalized)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

def save_index(index, meta_list, index_dir=INDEX_DIR):
    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))
    with open(os.path.join(index_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_list, f, ensure_ascii=False, indent=2)

def main():
    df = load_csv(CSV_PATH)
    # Build a single 'document' that is the answer + question meta
    docs = []
    for i, row in df.iterrows():
        docs.append({
            "id": int(i),
            "question": str(row["Question"]),
            "answer": str(row["Answer"])
        })
    texts_for_emb = [d["answer"] for d in docs]  # embed answers
    embeddings = build_embeddings(texts_for_emb)
    # normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    index = build_faiss_index(embeddings, dim=embeddings.shape[1])
    save_index(index, docs)
    print(f"Saved index with {len(docs)} docs in {INDEX_DIR}")

if __name__ == "__main__":
    main()
