"""REST API. Run: uvicorn api:app --port 8000"""
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent import Agent

state = {"sessions": {}}  # session_id -> message history (in-memory; use Redis/DB for multi-worker)


@asynccontextmanager
async def lifespan(app):
    state["agent"] = Agent()  # fails fast if the API key is missing
    yield


app = FastAPI(title="Warranty Claim Assistant", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    tool_calls: list[dict]
    audit_log: list[dict] = []


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    sid = req.session_id or str(uuid.uuid4())
    history = state["sessions"].setdefault(sid, [])
    mark = len(history)
    history.append({"role": "user", "content": req.message})
    try:
        reply, calls = state["agent"].chat(history)
    except Exception as e:
        del history[mark:]  # keep history valid for retry
        raise HTTPException(status_code=502, detail=f"LLM call failed: {getattr(e, 'message', type(e).__name__)}")
    audit_log = []
    for c in calls:  # the log travels in its own field, not duplicated inside tool_calls
        if c["tool"] == "recommend_claim_approval" and isinstance(c["result"], dict) and "audit_log" in c["result"]:
            audit_log = c["result"]["audit_log"]
            c["result"] = {k: v for k, v in c["result"].items() if k != "audit_log"}
    return ChatResponse(session_id=sid, reply=reply, tool_calls=calls, audit_log=audit_log)


@app.delete("/sessions/{session_id}")
def reset(session_id: str):
    state["sessions"].pop(session_id, None)
    return {"status": "cleared"}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
