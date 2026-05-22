from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import sys
from pathlib import Path
import uuid
import re

sys.path.append(str(Path(__file__).parent.parent))

from llama_cpp import Llama
from src.config import (
    MODEL_PATH,
    N_CTX,
    SYSTEM_RAG,
    SYSTEM_GENERAL,
    MAX_RESPONSE_TOKENS,
    TEMPERATURE,
    REPEAT_PENALTY,
)
from src.retriever import Retriever
from src.conversation import ConversationManager
from src.utils import is_small_talk, rewrite_query_with_context

retriever = None
llm = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, llm

    retriever = Retriever()
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=N_CTX,
        n_gpu_layers=-1,
        chat_format="zephyr",
        verbose=False,
    )
    yield


app = FastAPI(
    title="AI Tutor API",
    description="RAG-based Java OOP Tutor",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UI_DIR = Path(__file__).parent.parent / "ui"
app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    sources: List[str]
    session_id: str


class ResetRequest(BaseModel):
    session_id: str


_conv_manager_singleton = None


def get_conv_manager() -> ConversationManager:
    global _conv_manager_singleton
    if _conv_manager_singleton is None:
        _conv_manager_singleton = ConversationManager(max_token_limit=1000)
        _conv_manager_singleton.set_llm(llm)
    return _conv_manager_singleton


def get_or_create_conv_manager(session_id: str) -> ConversationManager:
    mgr = get_conv_manager()
    if session_id not in mgr.sessions:
        mgr.create_session(session_id)
    return mgr


# ENDPOINTS
@app.get("/")
async def root():
    """Serve το frontend HTML."""
    html_path = UI_DIR / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return {"status": "ok", "message": "AI Tutor API is running"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint."""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        conv_manager = get_or_create_conv_manager(session_id)

        user_message = request.message
        sources = []

        if is_small_talk(user_message):
            history_context = conv_manager.format_history_for_context(session_id)
            content = (
                f"{history_context}\n\n{user_message}"
                if history_context
                else user_message
            )
            messages = [
                {"role": "system", "content": SYSTEM_GENERAL},
                {"role": "user", "content": content},
            ]
        else:
            final_query = rewrite_query_with_context(
                llm, user_message, conv_manager, session_id
            )

            docs, metas = retriever.retrieve(final_query)

            if not docs:
                raise HTTPException(status_code=500, detail="Retrieval failed")

            ranked = retriever.rerank(final_query, docs, metas)
            selected, is_relevant = retriever.select_chunks(
                ranked, final_query, SYSTEM_RAG
            )

            if not is_relevant:
                return ChatResponse(
                    reply="Η ερώτηση δεν σχετίζεται με το υλικό Java OOP. Μπορείτε να τη διατυπώσετε διαφορετικά;",
                    sources=[],
                    session_id=session_id,
                )

            context, sources = retriever.build_context(selected)
            history_context = conv_manager.format_history_for_context(session_id)

            prompt_text = f"""{"Ιστορικό:" if history_context else ""}
{history_context}

ΑΠΟΣΠΑΣΜΑΤΑ:
{context}

ΕΡΩΤΗΣΗ: {user_message}

ΑΠΑΝΤΗΣΗ:"""

            messages = [
                {"role": "system", "content": SYSTEM_RAG},
                {"role": "user", "content": prompt_text},
            ]

        out = llm.create_chat_completion(
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_RESPONSE_TOKENS,
            repeat_penalty=REPEAT_PENALTY,
            stop=["<|user|>", "<|system|>", "<|assistant|>", "</s>"],
        )

        reply = out["choices"][0]["message"]["content"].strip()

        reply = re.sub(r"</?(?:assistant|user|system)>", "", reply)
        reply = reply.strip()

        conv_manager.add_turn(session_id, user_message, reply, sources)

        no_info_phrase = "δεν βρέθηκε σχετική πληροφορία"
        final_sources = [] if no_info_phrase in reply.lower() else sources

        return ChatResponse(reply=reply, sources=final_sources, session_id=session_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
async def reset_session(request: ResetRequest):
    mgr = get_conv_manager()
    mgr.clear_session(request.session_id)
    return {"status": "ok", "message": "Session reset"}


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "models_loaded": retriever is not None and llm is not None,
        "active_sessions": len(get_conv_manager().sessions),
    }
