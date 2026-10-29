import json
import os
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

from database import get_db_connection

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

SOCRATIC_SYSTEM_GUARDRAIL = """
You are the North Star Socratic Tutor.
STRICT PEDAGOGICAL GUARDRAILS:
1. NEVER give the student a direct final answer, full proof, or completed explanation.
2. ALWAYS respond by validating what the student got right, citing a specific detail from the Retrieved Slide Context, and asking ONE targeted, guiding question that forces active recall.
3. Keep responses concise (2 to 4 sentences maximum).
4. If the student clicks "Challenge Me", pose a concrete scenario or edge-case pop-quiz question directly from the Retrieved Slide Context.
"""


def get_genai_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing.")
    return genai.Client(api_key=api_key)


def _embed_texts(texts: List[str], client: genai.Client) -> List[List[float]]:
    """Generates vector embeddings using Gemini Embedding model."""
    res = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
    )
    return [emb.values for emb in res.embeddings]


def index_lecture_slides(file_hash: str, extracted_text: str) -> None:
    """
    Splits extracted lecture text into slide/paragraph chunks, computes embeddings,
    and stores them in the local SQLite vector index.
    """
    with get_db_connection() as conn:
        existing = conn.execute(
            "SELECT COUNT(*) as cnt FROM lecture_chunks WHERE file_hash = ?;",
            (file_hash,),
        ).fetchone()
        if existing and existing["cnt"] > 0:
            return

    raw_chunks = [
        c.strip()
        for c in extracted_text.split("### ")
        if c.strip()
    ]
    if not raw_chunks:
        raw_chunks = [extracted_text[i : i + 1000] for i in range(0, len(extracted_text), 1000)]

    chunks = [f"### {c}" if not c.startswith("#") else c for c in raw_chunks[:25]]

    client = get_genai_client()
    try:
        embeddings = _embed_texts(chunks, client)
    except Exception as exc:
        print(f"[VECTOR INDEX WARNING] Embedding fallback triggered: {exc}")
        embeddings = [[0.0] * 16 for _ in chunks]

    with get_db_connection() as conn:
        for idx, (chunk_text, vec) in enumerate(zip(chunks, embeddings)):
            conn.execute(
                """
                INSERT INTO lecture_chunks (file_hash, chunk_index, chunk_text, embedding_json)
                VALUES (?, ?, ?, ?);
                """,
                (file_hash, idx, chunk_text, json.dumps(vec)),
            )
        conn.execute(
            """
            INSERT OR IGNORE INTO socratic_memory (file_hash, knowledge_summary, messages_json)
            VALUES (?, 'Student has just begun exploring the lecture concepts.', '[]');
            """,
            (file_hash,),
        )


def retrieve_top_chunks(file_hash: str, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
    """Computes cosine similarity against local slide embeddings to retrieve top-k chunks."""
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT chunk_index, chunk_text, embedding_json FROM lecture_chunks WHERE file_hash = ?;",
            (file_hash,),
        ).fetchall()

    if not rows:
        return []

    client = get_genai_client()
    try:
        query_vec = np.array(_embed_texts([query], client)[0], dtype=np.float32)
    except Exception:
        return [{"chunk_index": rows[0]["chunk_index"] + 1, "chunk_text": rows[0]["chunk_text"]}]

    scored = []
    q_norm = np.linalg.norm(query_vec) or 1.0
    for r in rows:
        vec = np.array(json.loads(r["embedding_json"]), dtype=np.float32)
        if vec.shape != query_vec.shape:
            score = 0.0
        else:
            score = float(np.dot(query_vec, vec) / (q_norm * (np.linalg.norm(vec) or 1.0)))
        scored.append((score, r["chunk_index"] + 1, r["chunk_text"]))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {"chunk_index": item[1], "chunk_text": item[2]}
        for item in scored[:top_k]
    ]


def generate_socratic_response(
    file_hash: str, user_message: str, is_challenge_me: bool = False
) -> Dict[str, Any]:
    """
    Executes RAG retrieval + Asymmetric Memory Buffer (Knowledge State + Last 4 Messages).
    """
    with get_db_connection() as conn:
        mem_row = conn.execute(
            "SELECT knowledge_summary, messages_json FROM socratic_memory WHERE file_hash = ?;",
            (file_hash,),
        ).fetchone()

    knowledge_summary = (
        mem_row["knowledge_summary"]
        if mem_row
        else "Student has just started reviewing this lecture."
    )
    all_messages: List[Dict[str, Any]] = (
        json.loads(mem_row["messages_json"]) if mem_row else []
    )

    # ASYMMETRIC MEMORY BUFFER: Slice ONLY the last 4 messages
    recent_4_messages = all_messages[-4:]

    # Retrieve relevant slide chunks via local vector search
    search_query = "core lecture concepts quiz" if is_challenge_me else user_message
    retrieved_chunks = retrieve_top_chunks(file_hash, search_query, top_k=2)
    context_str = "\n\n".join(
        [f"[Slide Chunk {c['chunk_index']}]: {c['chunk_text']}" for c in retrieved_chunks]
    )

    prompt = f"""
    === BACKGROUND KNOWLEDGE STATE SUMMARY ===
    {knowledge_summary}

    === LAST 4 CONVERSATION TURNS (ASYMMETRIC BUFFER) ===
    {json.dumps(recent_4_messages, indent=2)}

    === RETRIEVED SLIDE CONTEXT (LOCAL VECTOR RAG) ===
    {context_str}

    === STUDENT INPUT ===
    {"[CHALLENGE ME TRIGGERED]: Give me a tough pop-quiz question from the slides!" if is_challenge_me else user_message}
    """

    client = get_genai_client()
    res = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SOCRATIC_SYSTEM_GUARDRAIL,
            temperature=0.3,
        ),
    )

    ai_reply = (res.text or "").strip()
    cited_slides = [f"Slide Chunk {c['chunk_index']}" for c in retrieved_chunks]

    # Append turn to persisted history
    all_messages.append({"sender": "user", "text": user_message})
    all_messages.append({"sender": "ai", "text": ai_reply, "citations": cited_slides})

    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO socratic_memory (file_hash, knowledge_summary, messages_json, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(file_hash) DO UPDATE SET
                messages_json = excluded.messages_json,
                updated_at = CURRENT_TIMESTAMP;
            """,
            (file_hash, knowledge_summary, json.dumps(all_messages)),
        )

    return {
        "reply": ai_reply,
        "citations": cited_slides,
        "knowledge_summary": knowledge_summary,
        "buffered_turn_count": len(recent_4_messages),
    }


def update_asymmetric_knowledge_state(file_hash: str) -> None:
    """
    Runs asynchronously in a background thread after each chat turn to compress
    the student's mastery state into a concise summary.
    """
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT knowledge_summary, messages_json FROM socratic_memory WHERE file_hash = ?;",
            (file_hash,),
        ).fetchone()

    if not row:
        return

    messages = json.loads(row["messages_json"])
    if len(messages) < 2:
        return

    client = get_genai_client()
    prompt = f"""
    Update the 2-sentence Knowledge State Summary of what this student understands
    and what they still need to master, based on their latest interaction.
    Previous Summary: {row['knowledge_summary']}
    Latest Turns: {json.dumps(messages[-4:])}
    """
    try:
        res = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt,
        )
        new_summary = (res.text or "").strip()
        if new_summary:
            with get_db_connection() as conn:
                conn.execute(
                    "UPDATE socratic_memory SET knowledge_summary = ?, updated_at = CURRENT_TIMESTAMP WHERE file_hash = ?;",
                    (new_summary, file_hash),
                )
    except Exception as exc:
        print(f"[ASYMMETRIC MEMORY WARNING] {exc}")