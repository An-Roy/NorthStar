# North Star 

North Star is a local-first, multi-modal AI academic study engine designed to ingest massive university lecture slide decks (PDFs) and dynamically route them into specialized pedagogical pipelines (Logic, Spatial, or Abstract). 

Built as a decoupled architecture using a Next.js frontend and a FastAPI Python backend, the system bypasses standard prompt-wrappers by leveraging the Gemini API's multimodal vision capabilities to extract flawless LaTeX, code, and spatial data. To ensure zero duplicate API costs and handle concurrent frontend polling, the state machine is anchored by a custom SQLite Write-Ahead Logging (WAL) caching layer with atomic Time-to-Live (TTL) locks.

### The Engineering & Architecture

Instead of dumping coordinate systems into a game loop, this backend solves three specific infrastructure constraints:

**1. O(1) Memory Cryptographic Hashing**
Before the system touches an API, uploaded PDFs are parsed in 64KB binary chunks using Python's native C-level `hashlib.file_digest`. This generates a SHA-256 fingerprint with a flat O(1) memory footprint—preventing Out-of-Memory (OOM) crashes on 2GB textbooks while providing an absolute source of truth for the local database cache.

**2. Concurrency-Safe SQLite (WAL) State Machine**
To prevent `sqlite3.OperationalError: database is locked` errors when the Next.js frontend asynchronously polls the backend for status updates, the database leverages Write-Ahead Logging (`PRAGMA journal_mode=WAL`). Reads and writes are completely unblocked. 
To prevent Time-of-Check to Time-of-Use (TOCTOU) race conditions—where a user double-clicks "Upload" and triggers duplicate LLM token burn—the database uses an atomic `INSERT OR IGNORE` upsert. The first request acquires a `PROCESSING` lock with a 300-second TTL; simultaneous requests instantly bounce back a cached wait-state.

**3. Deterministic Meta-Agent Routing**
Extracted raw LaTeX and Markdown (Phase 1) is passed to a Gemini 2.5 Flash Meta-Agent (Phase 2). To prevent the LLM from hallucinating free-text JSON, the router is bound by a strict Pydantic schema (`response_schema`). This guarantees the backend mathematically routes the lecture into one of three strict pipelines:
* **LOGIC:** (CS/Math) Outputs YouTube API queries and transcript validation criteria.
* **SPATIAL:** (Biology/Geography) Outputs visual generation prompts for pathway mapping.
* **ABSTRACT:** (Humanities) Outputs high-density audio podcast scripts.

### Tech Stack
* **Backend:** Python 3.11, FastAPI, `BackgroundTasks` (Thread Pool Orchestration)
* **Database:** SQLite3 (WAL Mode, Custom Connection Context Managers)
* **AI SDK:** `google-genai` (Gemini 2.5 Flash / Gemini Vision)
* **Data Validation:** Pydantic
* **Frontend (In Progress):** Next.js, React

### Local Development Setup

To run the backend ingestion pipeline locally on Windows 11:

1. **Initialize the Virtual Environment:**
`python -m venv venv`
`.\venv\Scripts\Activate.ps1`

2. **Install Dependencies:**
`pip install google-genai python-dotenv pydantic fastapi uvicorn`

3. **Configure Environment:**
Create a `.env` file in the root directory (encoded strictly in UTF-8 without BOM) and add your AI Studio API key:
`GEMINI_API_KEY=AIzaSy_YOUR_KEY_HERE`

4. **Execute the Pipeline Test:**
`python test_pipeline.py`

*Note: The test script fires a cold-start extraction on the first run, and instantly retrieves the Pydantic-routed payload from the local SQLite cache on the second run.*