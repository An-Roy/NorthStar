import json
import os
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, status, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from database import (
    acquire_extraction_lock,
    get_db_connection,
    init_db,
    release_failed_lock,
)
from generator import execute_phase3_generation, get_genai_client, MEDIA_DIR
from hasher import generate_file_hash
from ingest import process_lecture_pdf
from rag_engine import (
    generate_socratic_response,
    index_lecture_slides,
    update_asymmetric_knowledge_state,
)
from router import route_lecture_content

BASE_DIR = Path(__file__).resolve().parent
STAGING_DIR = BASE_DIR / "temp_uploads"
PDF_STORAGE_DIR = BASE_DIR / "stored_pdfs"

# Native Windows-safe ThreadPoolExecutor (replaces Celery/Redis daemon)
WORKER_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="northstar_worker")


@asynccontextmanager
async def lifespan(app: FastAPI):
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    PDF_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    init_db()
    yield
    WORKER_POOL.shutdown(wait=False)


app = FastAPI(
    title="North Star Pedagogical Engine API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------
class PipelineStatusResponse(BaseModel):
    file_hash: str
    status: str
    taxonomy_classification: Optional[str] = None
    routing_payload: Optional[Dict[str, Any]] = None
    raw_extracted_text: Optional[str] = None
    generated_media_path: Optional[str] = None
    pdf_url: Optional[str] = None
    knowledge_summary: Optional[str] = None
    updated_at: Optional[str] = None
    message: Optional[str] = None


class ChatRequest(BaseModel):
    file_hash: str
    message: str
    is_challenge_me: bool = False


class VerifyChallengeRequest(BaseModel):
    file_hash: str
    student_submission: str


# ---------------------------------------------------------------------------
# Asynchronous Pipeline Worker (Phases 1 -> 2 -> 3 -> 5 Vector Index)
# ---------------------------------------------------------------------------
def run_full_pipeline_worker(
    pdf_path: str,
    file_hash: str,
    resume_state: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Executes Phases 1, 2, 3, and 5 in the ThreadPoolExecutor:
    PROCESSING -> EXTRACTED -> ROUTED -> GENERATING -> COMPLETE
    """
    try:
        # Phase 1: Multimodal Extraction
        if resume_state and resume_state.get("raw_extracted_text"):
            extracted_text = resume_state["raw_extracted_text"]
        else:
            phase1_result = process_lecture_pdf(
                file_path=pdf_path,
                file_hash=file_hash,
                lock_already_acquired=True,
            )
            extracted_text = phase1_result["extracted_text"]

        # Phase 5 Prep: Build Local Vector Index on extracted slide paragraphs
        index_lecture_slides(file_hash, extracted_text)

        # Phase 2: Meta-Agent Taxonomy Routing
        if resume_state and resume_state.get("routing_payload"):
            routing_data = json.loads(resume_state["routing_payload"])
        else:
            routing_data = route_lecture_content(
                file_hash=file_hash,
                extracted_text=extracted_text,
            )

        # Phase 3: Downstream Media & Practice Sandbox Generation
        execute_phase3_generation(
            file_hash=file_hash,
            extracted_text=extracted_text,
            routing_data=routing_data,
        )
        print(f"[WORKER COMPLETE] {file_hash} transitioned to COMPLETE.")

    except Exception as exc:
        print(f"[WORKER ERROR] Pipeline failed for {file_hash}: {exc}")
        release_failed_lock(file_hash)


def _format_record(row: Dict[str, Any], message: Optional[str] = None) -> PipelineStatusResponse:
    parsed_payload = None
    if row.get("routing_payload"):
        try:
            parsed_payload = json.loads(row["routing_payload"])
        except json.JSONDecodeError:
            parsed_payload = {"raw": row["routing_payload"]}

    file_hash = row["file_hash"]
    knowledge_summary = None
    with get_db_connection() as conn:
        mem = conn.execute(
            "SELECT knowledge_summary FROM socratic_memory WHERE file_hash = ?;",
            (file_hash,),
        ).fetchone()
        if mem:
            knowledge_summary = mem["knowledge_summary"]

    return PipelineStatusResponse(
        file_hash=file_hash,
        status=row["status"],
        taxonomy_classification=row.get("taxonomy_classification"),
        routing_payload=parsed_payload,
        raw_extracted_text=row.get("raw_extracted_text"),
        generated_media_path=row.get("generated_media_path"),
        pdf_url=f"/pdf/{file_hash}",
        knowledge_summary=knowledge_summary,
        updated_at=row.get("updated_at"),
        message=message,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post(
    "/upload",
    response_model=PipelineStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def upload_lecture_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    temp_path = STAGING_DIR / f"{uuid.uuid4().hex}.pdf"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()

    file_hash = generate_file_hash(str(temp_path))
    permanent_pdf_path = PDF_STORAGE_DIR / f"{file_hash}.pdf"
    if not permanent_pdf_path.exists():
        shutil.copyfile(temp_path, permanent_pdf_path)
    temp_path.unlink(missing_ok=True)

    lock_acquired, existing_record = acquire_extraction_lock(file_hash)

    if not lock_acquired and existing_record is not None:
        current_status = existing_record["status"]

        # Upgrade older 'EXTRACTED' or 'ROUTED' records into full Phase 3 'COMPLETE' records
        if current_status in ("EXTRACTED", "ROUTED"):
            WORKER_POOL.submit(
                run_full_pipeline_worker,
                str(permanent_pdf_path),
                file_hash,
                existing_record,
            )
            return _format_record(
                existing_record,
                message=f"Resuming Phase 3 Media & RAG Indexing from {current_status}...",
            )

        return _format_record(existing_record, message="Cache hit.")

    # Submit full Phase 1 -> 2 -> 3 -> 5 job to ThreadPoolExecutor
    WORKER_POOL.submit(
        run_full_pipeline_worker,
        str(permanent_pdf_path),
        file_hash,
        None,
    )

    return PipelineStatusResponse(
        file_hash=file_hash,
        status="PROCESSING",
        pdf_url=f"/pdf/{file_hash}",
        message="Job queued in ThreadPoolExecutor.",
    )


@app.get("/status/{file_hash}", response_model=PipelineStatusResponse)
def get_pipeline_status(file_hash: str):
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT * FROM lecture_cache WHERE file_hash = ?;",
            (file_hash,),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Record not found.")

    return _format_record(dict(row))


@app.get("/pdf/{file_hash}")
def serve_original_pdf(file_hash: str):
    """Serves the original uploaded PDF for the Left Pane of the Dual-Pane UI."""
    pdf_path = PDF_STORAGE_DIR / f"{file_hash}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Original PDF not found on disk.")
    return FileResponse(pdf_path, media_type="application/pdf")


@app.get("/media/{filename}")
def serve_generated_media(filename: str):
    """Streams generated Phase 3 MP4 video, WAV studio audio, or MP3 podcast files."""
    file_path = MEDIA_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Generated media file not found.")
    if filename.endswith(".mp4"):
        media_type = "video/mp4"
    elif filename.endswith(".wav"):
        media_type = "audio/wav"
    else:
        media_type = "audio/mpeg"
    return FileResponse(file_path, media_type=media_type)


@app.post("/chat")
def socratic_rag_chat(req: ChatRequest, background_tasks: BackgroundTasks):
    """
    Phase 5 Endpoint: Retrieves local vector slide chunks, applies the 4-message
    Asymmetric Memory Buffer + Socratic Guardrail, and schedules background state compression.
    """
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT raw_extracted_text FROM lecture_cache WHERE file_hash = ?;",
            (req.file_hash,),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Lecture not found.")

    # Ensure vector index exists even for older cached files
    if row["raw_extracted_text"]:
        index_lecture_slides(req.file_hash, row["raw_extracted_text"])

    result = generate_socratic_response(
        file_hash=req.file_hash,
        user_message=req.message,
        is_challenge_me=req.is_challenge_me,
    )

    # Update the summarized knowledge state asynchronously
    background_tasks.add_task(update_asymmetric_knowledge_state, req.file_hash)
    return result


@app.post("/verify-challenge")
def verify_logic_sandbox(req: VerifyChallengeRequest):
    """
    Phase 4 Active Practice Sandbox Gatekeeper: Evaluates whether the student's
    math proof or code solution passes the rubric to unlock progression.
    """
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT routing_payload FROM lecture_cache WHERE file_hash = ?;",
            (req.file_hash,),
        ).fetchone()
    if not row or not row["routing_payload"]:
        raise HTTPException(status_code=404, detail="Routing payload not found.")

    payload = json.loads(row["routing_payload"])
    sandbox = payload.get("practice_sandbox", {})

    client = get_genai_client()
    prompt = f"""
    Evaluate the student's submission for this university LOGIC challenge.
    Question: {sandbox.get('question_markdown')}
    Rubric: {sandbox.get('grading_rubric')}
    Student Submission: {req.student_submission}

    Respond in JSON with:
    - "passed": boolean (true if mathematically/logically sound, false otherwise)
    - "feedback": string (if false, give a Socratic hint without revealing the answer; if true, congratulate and explain why it works)
    """
    res = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )
    return json.loads(res.text)