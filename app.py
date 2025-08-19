import os
import uuid
import time
import json
import asyncio
from typing import List, Dict, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from tools import persist_uploads, persist_base64_files
from agent import build_crew

# ---- Environment setup ----
TMP_DIR = "/tmp/.mem0"
PERSIST_DIR = "/tmp/.embedchain"
STORAGE_DIR = "/tmp/crewai_storage"

os.environ["EMBEDCHAIN_CONFIG_DIR"] = PERSIST_DIR
os.environ["MEM0_DIR"] = TMP_DIR
os.environ["CREWAI_STORAGE_PATH"] = STORAGE_DIR
os.environ["XDG_DATA_HOME"] = STORAGE_DIR
os.environ["HOME"] = "/tmp" 
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(os.path.join("/tmp", ".local", "share"), exist_ok=True) 
# ---- Runtime configuration ----
APP_HARD_TIMEOUT = int(os.getenv("APP_HARD_TIMEOUT", "170"))
RUN_ROOT = "/tmp/agent_runs"
os.makedirs(RUN_ROOT, exist_ok=True)

# ---- FastAPI app ----
app = FastAPI(title="Agentic Analyst API (FastAPI + CrewAI + Gemini)")

class AskResponse(BaseModel):
    answer: str
    elapsed_sec: float
    artifacts: Optional[dict] = None

@app.post("/api/", response_model=AskResponse)
async def run_agent(
    request: Request,
    questions_file: UploadFile = File(..., alias="questions.txt"),
    all_files: List[UploadFile] = File(default=[]),  # catch-all bucket
    base64_files: Optional[str] = Form(None)
):
    # ---- Per-request working dir ----
    run_id = str(uuid.uuid4())
    workdir = os.path.join(RUN_ROOT, run_id)
    os.makedirs(workdir, exist_ok=True)

    # ---- Persist questions.txt ----
    qpath = os.path.join(workdir, "questions.txt")
    with open(qpath, "wb") as f:
        f.write(await questions_file.read())

    with open(qpath, "r", encoding="utf-8", errors="replace") as f:
        question = f.read().strip()

    if not question:
        raise HTTPException(status_code=400, detail="questions.txt must not be empty")

    # ---- Persist all other uploaded files ----
    saved_files: Dict[str, str] = {}
    if all_files:
        saved_files.update(persist_uploads(all_files, workdir))

    # ---- Persist base64 files ----
    if base64_files:
        try:
            base64_dict = json.loads(base64_files)
            if not isinstance(base64_dict, dict):
                raise ValueError
            saved_files.update(persist_base64_files(base64_dict, workdir))
        except Exception:
            raise HTTPException(status_code=400, detail="base64_files must be a valid JSON object")

    # ---- Ensure questions.txt is in saved_files ----
    saved_files["questions.txt"] = qpath

    # ---- Build CrewAI agent ----
    crew = build_crew(workdir=workdir, saved_files=saved_files)

    # ---- Run agent ----
    start = time.time()
    try:
        async with asyncio.timeout(APP_HARD_TIMEOUT):
            def _kick():
                return crew.kickoff(inputs={"question": question})
            result = await asyncio.to_thread(_kick)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Computation exceeded the 3-minute time budget")

    elapsed = time.time() - start
    answer = str(result.raw) if hasattr(result, "raw") else str(result)

    return JSONResponse({
        "answer": answer,
        "elapsed_sec": round(elapsed, 2),
        "artifacts": {
            "workdir": workdir,
            "files": list(saved_files.values())
        }
    })
