import os
import time
from typing import Dict, Any
from google import genai
from google.genai import types
from dotenv import load_dotenv

from database import (
    init_db,
    acquire_extraction_lock,
    save_extracted_text,
    release_failed_lock,
)
from hasher import generate_file_hash

from pathlib import Path
from dotenv import load_dotenv

# Resolve the exact directory where this .py file lives so working-directory shifts never break .env loading
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

def get_genai_client() -> genai.Client:
    """Lazily initializes and validates the Gemini API client."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "YOUR_ACTUAL_KEY_HERE":
        raise RuntimeError(
            f"GEMINI_API_KEY is missing or invalid. Checked path: {ENV_PATH}"
        )
    return genai.Client(api_key=api_key)



EXTRACTION_PROMPT = """
Extract all text from this lecture slide deck chronologically slide-by-slide.
1. Preserve all mathematical proofs, symbolic logic, and formulas using strict LaTeX formatting ($...$ for inline, $$...$$ for block).
2. Preserve all code snippets in fenced markdown code blocks with the language specified.
3. Describe any critical spatial diagrams (e.g., biological pathways, memory state diagrams) inside [DIAGRAM: ...] tags.
Do not summarize or omit details. Output the complete structural content.
"""


def process_lecture_pdf(
    file_path: str,
    file_hash: str | None = None,
    lock_already_acquired: bool = False,
) -> Dict[str, Any]:
    """
    Executes Phase 1: Hashes PDF, acquires or reclaims atomic TTL DB lock,
    and extracts text/LaTeX via Gemini Vision if uncached.
    """
    init_db()
    if file_hash is None:
        file_hash = generate_file_hash(file_path)

    # Step 1: Attempt atomic lock acquisition unless the HTTP layer already claimed it
    if not lock_already_acquired:
        lock_acquired, existing_record = acquire_extraction_lock(file_hash)

        if not lock_acquired and existing_record is not None:
            status = existing_record["status"]
            if status == "PROCESSING":
                return {
                    "file_hash": file_hash,
                    "status": "PROCESSING",
                    "message": "Extraction is actively running (within TTL window).",
                }
            return {
                "file_hash": file_hash,
                "status": status,
                "extracted_text": existing_record["raw_extracted_text"],
                "taxonomy": existing_record["taxonomy_classification"],
                "routing_payload": existing_record["routing_payload"],
                "media_path": existing_record["generated_media_path"],
            }
    
    client = get_genai_client()

    # Step 2: Lock acquired (fresh or reclaimed from zombie). Call Gemini Files & Vision API.
    uploaded_pdf = None
    try:
        print(f"[LOCK ACQUIRED] Uploading {file_path} to Gemini File API...")
        uploaded_pdf = client.files.upload(
            file=file_path,
            config={"mime_type": "application/pdf"}
        )

# Poll file state using exponential backoff to avoid 429 Rate Limits
        poll_start = time.time()
        backoff_seconds = 2  # Start with a 2-second wait
        
        while uploaded_pdf.state.name == "PROCESSING":
            if time.time() - poll_start > 120:
                raise TimeoutError("Gemini File API timed out while staging the PDF.")
            
            print(f"Waiting {backoff_seconds}s for Google to process the PDF...")
            time.sleep(backoff_seconds)
            
            # Fetch the updated state
            uploaded_pdf = client.files.get(name=uploaded_pdf.name)
            
            # Double the wait time for the next loop (max 10 seconds)
            backoff_seconds = min(backoff_seconds * 2, 10)

        if uploaded_pdf.state.name == "FAILED":
            raise RuntimeError("Gemini File API failed to process the PDF.")

        # Execute multimodal extraction with a 240-second (240,000 ms) HTTP timeout
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=[uploaded_pdf, EXTRACTION_PROMPT],
            config=types.GenerateContentConfig(
                http_options=types.HttpOptions(timeout=240_000)
            ),
        )

        extracted_text = response.text
        save_extracted_text(file_hash, extracted_text)

        return {
            "file_hash": file_hash,
            "status": "EXTRACTED",
            "extracted_text": extracted_text,
            "taxonomy": None,
            "routing_payload": None,
            "media_path": None,
        }

    except Exception as e:
        release_failed_lock(file_hash)
        raise RuntimeError(f"Extraction pipeline failed for {file_hash}: {e}") from e

    finally:
        if uploaded_pdf is not None:
            try:
                client.files.delete(name=uploaded_pdf.name)
            except Exception as cleanup_err:
                print(f"[WARNING] Failed to delete remote file {uploaded_pdf.name}: {cleanup_err}")