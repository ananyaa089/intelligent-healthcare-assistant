from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR = "index_data"
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

# ---------------------- CORS FIX ----------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # allow all origins (frontend)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    top_k: int = TOP_K

@app.on_event("startup")
def load_index():
    global index, meta, embedder

    idx_path = os.path.join(INDEX_DIR, "faiss.index")
    meta_path = os.path.join(INDEX_DIR, "meta.json")

    if not os.path.exists(idx_path) or not os.path.exists(meta_path):
        raise RuntimeError("Index files not found. Run preprocess_and_index.py first.")

    index = faiss.read_index(idx_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    embedder = SentenceTransformer(EMB_MODEL_NAME)

@app.get("/health")
def health():
    return {"status": "ok", "docs": len(meta)}

@app.post("/ask")
def ask(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query is empty")

    q_emb = embedder.encode([req.query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)

    D, I = index.search(q_emb, req.top_k)

    results = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        doc = meta[idx]
        results.append({
            "id": doc["id"],
            "question": doc["question"],
            "answer": doc["answer"],
            "score": float(score)
        })

    if results:
        return {"query": req.query, "answer": results[0]["answer"], "sources": results}
    
    return {"query": req.query, "answer": "No results found.", "sources": []}
