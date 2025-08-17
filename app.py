from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional
import asyncio, os, time, uuid, json

from tools import persist_uploads, persist_base64_files
from agent import build_crew

APP_HARD_TIMEOUT = int(os.getenv("APP_HARD_TIMEOUT", "170"))  # < 3 minutes
RUN_ROOT = os.getenv("RUN_ROOT", "/tmp/agent_runs")
os.makedirs(RUN_ROOT, exist_ok=True)

app = FastAPI(title="Agentic Analyst API (FastAPI + CrewAI + Gemini)")

class AskResponse(BaseModel):
    answer: str
    elapsed_sec: float
    artifacts: Optional[dict] = None

@app.post("/api/", response_model=AskResponse)
async def run_agent(
    files: List[UploadFile] = File(...),
    base64_files: Optional[str] = Form(None)  # JSON string, optional
):
    # Ensure questions.txt exists
    qfile = next((f for f in files if f.filename.lower() == "questions.txt"), None)
    if not qfile:
        raise HTTPException(status_code=400, detail="questions.txt is required in multipart form-data")

    question = qfile.file.read().decode("utf-8", errors="replace").strip()
    other_files = [f for f in files if f is not qfile]

    # Parse base64_files JSON string if provided
    base64_dict: Dict[str, str] = {}
    if base64_files:
        try:
            base64_dict = json.loads(base64_files)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="base64_files must be a valid JSON object")

    # Create per-request working dir
    run_id = str(uuid.uuid4())
    workdir = os.path.join(RUN_ROOT, run_id)
    os.makedirs(workdir, exist_ok=True)

    # Persist uploaded files and optional base64 files
    saved_files = {}
    if other_files:
        saved_files.update(persist_uploads(other_files, workdir))
    if base64_dict:
        saved_files.update(persist_base64_files(base64_dict, workdir))

    # Build CrewAI agent
    crew = build_crew(workdir=workdir, saved_files=saved_files)

    start = time.time()
    try:
        async with asyncio.timeout(APP_HARD_TIMEOUT):
            # kickoff is synchronous; run in thread
            def _kick():
                return crew.kickoff(inputs={"question": question})
            result = await asyncio.to_thread(_kick)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Computation exceeded the 3-minute time budget")

    elapsed = time.time() - start
    answer = str(getattr(result, "raw", result))

    return JSONResponse({
        "answer": answer,
        "elapsed_sec": round(elapsed, 2),
        "artifacts": {
            "workdir": workdir,
            "files": list(saved_files.values())
        }
    })
