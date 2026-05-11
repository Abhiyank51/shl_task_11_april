from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

from app.schemas import ChatRequest
from app.agent import run_agent

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm all indexes at startup so first request is fast
    try:
        from app.catalog_loader import get_catalog
        get_catalog()
        print("[startup] Catalog loaded.")
    except Exception as e:
        raise RuntimeError(f"Catalog load failed: {e}")

    try:
        from app.retriever import get_tfidf_index
        get_tfidf_index()
        print("[startup] TF-IDF index built.")
    except Exception as e:
        print(f"[startup] TF-IDF index warning: {e}")

    try:
        from app.vector_store import ensure_index_built
        ensure_index_built()
        print("[startup] FAISS vector index built.")
    except Exception as e:
        print(f"[startup] Vector index warning (non-fatal): {e}")

    yield

app = FastAPI(title="SHL Assessment Recommender API", lifespan=lifespan)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    if not request.messages:
        return JSONResponse(
            status_code=200,
            content={
                "reply": "Please provide a conversation message describing the role, skills, or SHL assessment need.",
                "recommendations": [],
                "end_of_conversation": False
            }
        )
        
    try:
        messages_dict = [{"role": m.role, "content": m.content} for m in request.messages]
        response_dict = run_agent(messages_dict)
        return response_dict
    except Exception as e:
        print(f"Error processing chat: {e}")
        return JSONResponse(
            status_code=200,
            content={
                "reply": "I could not process that request safely. Please rephrase the SHL assessment requirement.",
                "recommendations": [],
                "end_of_conversation": False
            }
        )
