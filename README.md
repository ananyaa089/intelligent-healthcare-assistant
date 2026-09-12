# Intelligent Healthcare Assistant

A retrieval-augmented generation (RAG) system that answers medical questions by
combining semantic search with LLM-based synthesis, built with safety guardrails
appropriate for a medical domain.

**Live demo:** https://intelligent-healthcare-assistant.vercel.app/

> Hosted on a free-tier backend that sleeps after 15 minutes of inactivity - 
> the first request may take 30–50s to wake it up. Give it a moment on first load.

## What it does

Given a medical question, the system:
1. Embeds the query and searches a FAISS index of ~16,000 Q&A pairs from the
   [MedQuAD](https://github.com/abachaa/MedQuAD) dataset for the closest matches
2. Checks a confidence threshold on the retrieval score, if nothing matches well
   enough, it returns a safe fallback ("consult a healthcare professional") instead
   of guessing
3. If confidence is high enough, feeds the retrieved context into Llama 3.3 70B
   (via Groq) to generate a synthesized, plain-language answer grounded in that
   context, rather than just returning the raw closest match
4. Returns the answer along with its sources and a confidence label, so the response
   isn't a black box

The system prompt explicitly restricts the model from providing diagnoses, dosage
instructions, or personalized treatment advice.

## Architecture

```
User query
   │
   ▼
React frontend (Vite)
   │  POST /ask
   ▼
FastAPI backend
   │
   ├─► sentence-transformers (all-MiniLM-L6-v2) → embed query
   ├─► FAISS index search → top-k matches + similarity scores
   ├─► confidence check → fallback if below threshold
   └─► Groq (Llama 3.3 70B) → generate grounded answer
   │
   ▼
Response: { answer, sources, confidence, disclaimer }
```

## Tech stack

- **Backend:** FastAPI, FAISS, sentence-transformers, Groq API
- **Frontend:** React, Vite
- **Deployment:** Docker, Render (backend), Vercel (frontend)

## Why the confidence threshold matters

For a general-purpose chatbot, always generating *some* answer is fine. For a medical
one, confidently answering a question the underlying dataset has no good match for is
actively harmful. The threshold check exists specifically to catch that case and defer
to "talk to a professional" instead of letting the LLM improvise from weak context.

## Disclaimer

This project provides general health information for educational purposes and is not
a substitute for professional medical advice, diagnosis, or treatment.
