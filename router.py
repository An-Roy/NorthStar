import os
import json
from typing import Literal, List, Dict, Any
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

from database import save_routing_decision

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



# 1. Define the strict Pydantic contract
class TaxonomyRoute(BaseModel):
    classification: Literal["LOGIC", "SPATIAL", "ABSTRACT"] = Field(
        description="The strict pedagogical route: LOGIC (math, proofs, algorithms, coding), SPATIAL (biology, anatomy, physical systems), or ABSTRACT (philosophy, history, pure humanities)."
    )
    confidence: float = Field(
        description="Confidence score between 0.0 and 1.0."
    )
    core_concepts: List[str] = Field(
        description="3 to 5 specific technical concepts or theorems extracted from the slides."
    )
    youtube_search_queries: List[str] = Field(
        description="Populated ONLY if classification is LOGIC. 3 syllabus-specific YouTube queries targeting step-by-step university proofs or code execution."
    )
    transcript_validation_criteria: List[str] = Field(
        description="Populated ONLY if classification is LOGIC. Exact mathematical steps, syntax, or theorems a YouTube transcript must contain to pass validation."
    )
    spatial_explainer_prompt: str = Field(
        description="Populated ONLY if classification is SPATIAL. A chronological visual & audio explainer script describing physical structures and state changes."
    )
    abstract_podcast_script: str = Field(
        description="Populated ONLY if classification is ABSTRACT. A high-density conversational audio script unpacking the theoretical arguments."
    )


ROUTER_SYSTEM_PROMPT = """
You are the Meta-Agent Taxonomy Router for an academic learning engine.
Analyze the extracted lecture slides and classify them into exactly ONE pedagogical pipeline:

1. LOGIC: Computer Science (Python, data structures, algorithms), Mathematics (MAT102 proofs, induction, calculus, linear algebra).
   - Must populate `youtube_search_queries` and `transcript_validation_criteria`. Leave `spatial_explainer_prompt` and `abstract_podcast_script` as empty strings.

2. SPATIAL: Biology for Health Sciences, Chemistry, Anatomy, Astronomy, Physical Geography.
   - Must populate `spatial_explainer_prompt` with a scene-by-scene visual diagram and narration prompt. Leave LOGIC and ABSTRACT fields empty.

3. ABSTRACT: Philosophy, History, Ethics, Literature, Pure Theory.
   - Must populate `abstract_podcast_script` with a structured single-voice narration script. Leave LOGIC and SPATIAL fields empty.
"""


def route_lecture_content(file_hash: str, extracted_text: str) -> Dict[str, Any]:
    """
    Classifies the extracted lecture text using Gemini 2.5 Flash with a strict
    Pydantic response schema and persists the routing payload to SQLite.
    """
    client = get_genai_client()
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=extracted_text,
        config=types.GenerateContentConfig(
            system_instruction=ROUTER_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=TaxonomyRoute,
            temperature=0.1,  # Low temperature for deterministic routing
        ),
    )

    # Guaranteed to parse cleanly into our Pydantic schema
    routing_data = json.loads(response.text)
    classification = routing_data["classification"]

    # Persist to SQLite so Phase 3 workers can read it
    save_routing_decision(
        file_hash=file_hash,
        classification=classification,
        routing_payload=json.dumps(routing_data),
    )

    return routing_data