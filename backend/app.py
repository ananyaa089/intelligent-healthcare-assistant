from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import faiss
import numpy as np
from fastembed import TextEmbedding
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv(Path(__file__).resolve().parent / ".env")

INDEX_DIR = "index_data"
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.45"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

SYSTEM_PROMPT = """You are a medical information assistant. You answer ONLY using the
context passages provided below, which come from a vetted medical Q&A dataset (MedQuAD).

Rules you must follow strictly:
1. Do not use any medical knowledge outside the provided context.
2. If the context does not clearly answer the question, say you don't have reliable
   information on this and recommend consulting a healthcare professional. Do not guess.
3. Never provide a diagnosis, dosage instructions, or personalized treatment advice.
   Reframe such questions toward general educational information and recommend a doctor.
4. Keep answers clear, concise, and in plain language a non-expert can understand.
5. This is general health information, not a substitute for professional medical advice.
"""

app = FastAPI(title="Intelligent Healthcare Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


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

    embedder = TextEmbedding(model_name=EMB_MODEL_NAME)

    if groq_client is None:
        print("WARNING: GROQ_API_KEY not set. /ask will fall back to raw retrieval "
              "(no generation) until it's configured.")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "docs": len(meta),
        "generation_enabled": groq_client is not None,
    }


def retrieve(query: str, top_k: int):
    q_emb = np.array(list(embedder.embed([query])), dtype=np.float32)
    faiss.normalize_L2(q_emb)

    D, I = index.search(q_emb, top_k)

    results = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        doc = meta[idx]
        results.append({
            "id": doc["id"],
            "question": doc["question"],
            "answer": doc["answer"],
            "score": float(score),
        })
    return results


def build_context(results):
    blocks = []
    for i, r in enumerate(results):
        blocks.append(f"[Source {i+1}] Q: {r['question']}\nA: {r['answer']}")
    return "\n\n".join(blocks)


def generate_answer(query: str, results: list) -> str:
    context = build_context(results)
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer using only the context above."

    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=400,
    )
    return completion.choices[0].message.content.strip()


FALLBACK_MESSAGE = (
    "I don't have reliable information to answer that confidently. "
    "Please consult a licensed healthcare professional for guidance on this."
)


@app.post("/ask")
def ask(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query is empty")

    results = retrieve(req.query, req.top_k)

    if not results:
        return {
            "query": req.query,
            "answer": FALLBACK_MESSAGE,
            "sources": [],
            "confidence": "none",
            "disclaimer": "This is general health information, not medical advice.",
        }

    top_score = results[0]["score"]
    confident = top_score >= CONFIDENCE_THRESHOLD

    if not confident:
        return {
            "query": req.query,
            "answer": FALLBACK_MESSAGE,
            "sources": results,
            "confidence": "low",
            "top_score": top_score,
            "disclaimer": "This is general health information, not medical advice.",
        }

    if groq_client is None:
        answer = results[0]["answer"]
    else:
        try:
            answer = generate_answer(req.query, results)
        except Exception as e:
            answer = results[0]["answer"]
            print(f"Generation failed, falling back to retrieval: {e}")

    return {
        "query": req.query,
        "answer": answer,
        "sources": results,
        "confidence": "high" if top_score >= 0.6 else "medium",
        "top_score": top_score,
        "disclaimer": "This is general health information, not medical advice. "
                       "Please consult a healthcare professional for personal medical concerns.",
    }
