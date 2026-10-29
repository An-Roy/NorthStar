import base64
import json
import os
import re
import time
import wave
from pathlib import Path
from typing import Dict, Any, List

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import types
from gtts import gTTS
from pydantic import BaseModel, Field
from youtube_transcript_api import YouTubeTranscriptApi

from database import mark_generating_media, save_generated_media

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

MEDIA_DIR = Path(__file__).resolve().parent / "generated_media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)


def get_genai_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing.")
    return genai.Client(api_key=api_key)


def _clean_text_for_tts(text: str) -> str:
    """
    Strips Markdown symbols AND speaker prefixes ('Andrew:', 'Ava:', '[ANDREW]:')
    so single-voice fallbacks never read speaker names out loud.
    """
    cleaned = re.sub(r"(?m)^\s*\[?(?:ANDREW|AVA|Andrew|Ava)\]?\s*:\s*", "", text)
    cleaned = re.sub(r"\[(?:ANDREW|AVA|Andrew|Ava)\]\s*:\s*", "", cleaned)
    cleaned = re.sub(r"[*_#`~>]", "", cleaned)
    cleaned = re.sub(r"\[Part\s+\d+.*?\]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _split_dialogue_into_batches(text: str, max_chars: int = 3800) -> List[str]:
    """
    Splits a multi-speaker script along speaker turns into ~3,800-character batches
    so each Gemini TTS call stays well within output audio token limits.
    """
    # Normalize any [ANDREW]: / [AVA]: tags to 'Andrew:' / 'Ava:'
    normalized = re.sub(r"\[ANDREW\]\s*:", "Andrew:", text, flags=re.IGNORECASE)
    normalized = re.sub(r"\[AVA\]\s*:", "Ava:", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"[*_#`~>]", "", normalized)

    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    batches: List[str] = []
    current_batch: List[str] = []
    current_len = 0

    for line in lines:
        if current_len + len(line) + 2 > max_chars and current_batch:
            batches.append("\n".join(current_batch))
            current_batch = [line]
            current_len = len(line)
        else:
            current_batch.append(line)
            current_len += len(line) + 2

    if current_batch:
        batches.append("\n".join(current_batch))

    return batches


def _synthesize_gemini_multispeaker_wav(
    script_text: str, output_wav_path: Path, client: genai.Client
) -> bool:
    """
    Uses Google's native 'gemini-2.5-flash-preview-tts' Multi-Speaker engine
    (NotebookLM voices 'Puck' for Andrew and 'Kore' for Ava) over your existing
    GEMINI_API_KEY and saves a 24kHz studio WAV file.
    """
    batches = _split_dialogue_into_batches(script_text, max_chars=3800)
    if not batches:
        return False

    combined_pcm = bytearray()

    for idx, batch in enumerate(batches):
        print(
            f"[GEMINI MULTI-SPEAKER TTS] Synthesizing studio audio segment {idx + 1}/{len(batches)}..."
        )
        tts_prompt = (
            "TTS the following educational podcast conversation naturally, "
            "warmly, and expressively:\n\n" + batch
        )

        segment_bytes = None
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash-preview-tts",
                    contents=tts_prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                                speaker_voice_configs=[
                                    types.SpeakerVoiceConfig(
                                        speaker="Andrew",
                                        voice_config=types.VoiceConfig(
                                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                                voice_name="Puck"
                                            )
                                        ),
                                    ),
                                    types.SpeakerVoiceConfig(
                                        speaker="Ava",
                                        voice_config=types.VoiceConfig(
                                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                                voice_name="Kore"
                                            )
                                        ),
                                    ),
                                ]
                            )
                        ),
                    ),
                )

                for part in response.candidates[0].content.parts:
                    if part.inline_data and part.inline_data.data:
                        raw_data = part.inline_data.data
                        segment_bytes = (
                            base64.b64decode(raw_data)
                            if isinstance(raw_data, str)
                            else raw_data
                        )
                        break

                if segment_bytes:
                    break
            except Exception as err:
                print(
                    f"[GEMINI TTS RETRY {attempt + 1}/2] Segment {idx + 1} warning: {err}"
                )
                time.sleep(4)

        if not segment_bytes:
            return False

        combined_pcm.extend(segment_bytes)
        if idx + 1 < len(batches):
            time.sleep(2)

    if len(combined_pcm) == 0:
        return False

    # Write 24kHz, 16-bit mono PCM audio into a standard WAV container
    with wave.open(str(output_wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(bytes(combined_pcm))

    print(
        f"[GEMINI MULTI-SPEAKER TTS SUCCESS] Studio 2-host audio saved to {output_wav_path.name}"
    )
    return True


def _synthesize_audio_resilient(
    text: str, file_hash: str, suffix: str, client: genai.Client
) -> str:
    """
    Synthesizes podcast/narration audio using:
    1. Gemini 2.5 Flash Native Multi-Speaker Studio TTS (.wav)
    2. Emergency gTTS Fallback with speaker names stripped (.mp3)
    """
    wav_filename = f"{file_hash}_{suffix}.wav"
    wav_path = MEDIA_DIR / wav_filename

    if _synthesize_gemini_multispeaker_wav(text, wav_path, client):
        return f"/media/{wav_filename}"

    # Emergency Fallback: gTTS with speaker labels stripped so it never reads "Andrew" or "Ava"
    mp3_filename = f"{file_hash}_{suffix}.mp3"
    mp3_path = MEDIA_DIR / mp3_filename
    clean_text = _clean_text_for_tts(
        text or "Welcome to the North Star lecture synthesis podcast."
    )
    try:
        tts = gTTS(text=clean_text, lang="en", slow=False)
        tts.save(str(mp3_path))
        print(f"[TTS FALLBACK] Audio synthesized via gTTS at {mp3_filename}")
        return f"/media/{mp3_filename}"
    except Exception as gtts_err:
        print(f"[TTS ERROR] All TTS engines failed: {gtts_err}")
        return ""


# ---------------------------------------------------------------------------
# Path A (LOGIC): Schema for Active Practice Sandbox & Transcript Validation
# ---------------------------------------------------------------------------
class LogicSandboxChallenge(BaseModel):
    question_markdown: str = Field(
        description="A rigorous university-level math proof step or coding challenge based on the lecture slides."
    )
    starter_template: str = Field(
        description="Starter equation or code skeleton for the student to complete."
    )
    hint: str = Field(
        description="A subtle conceptual hint that does not give away the answer."
    )
    grading_rubric: str = Field(
        description="Exact mathematical or logical steps required for the student's submission to pass."
    )


def _search_youtube_video_ids(query: str, max_results: int = 3) -> List[str]:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        resp = httpx.get(
            "https://www.youtube.com/results",
            params={"search_query": query},
            headers=headers,
            timeout=10.0,
        )
        video_ids = re.findall(r'"videoId":"([a-zA-Z0-9_-]{11})"', resp.text)
        unique_ids = []
        for vid in video_ids:
            if vid not in unique_ids:
                unique_ids.append(vid)
            if len(unique_ids) >= max_results:
                break
        return unique_ids
    except Exception as exc:
        print(f"[YOUTUBE SEARCH WARNING] {exc}")
        return []


def _fetch_and_validate_youtube(
    queries: List[str], criteria: List[str], client: genai.Client
) -> Dict[str, Any]:
    ytt_api = YouTubeTranscriptApi()
    validated_playlist: List[Dict[str, str]] = []
    used_video_ids = set()

    for idx, query in enumerate(queries):
        candidate_ids = _search_youtube_video_ids(query, max_results=4)
        candidate_ids = [vid for vid in candidate_ids if vid not in used_video_ids]

        if not candidate_ids:
            continue

        selected_vid = candidate_ids[0]
        validation_summary = f"Matched syllabus module {idx + 1} query: '{query}'."

        for vid in candidate_ids[:3]:
            try:
                transcript_list = ytt_api.fetch(vid)
                full_transcript = " ".join([s.text for s in transcript_list])[:10000]

                val_prompt = f"""
                Topic Query: {query}
                Validation Criteria: {json.dumps(criteria)}
                Transcript Excerpt: {full_transcript[:5000]}
                In 1 concise sentence, confirm which mathematical steps, proofs, or code concepts from the criteria this video covers.
                """
                res = client.models.generate_content(
                    model="gemini-3.1-flash-lite",
                    contents=val_prompt,
                )
                selected_vid = vid
                validation_summary = (res.text or "").strip()
                break
            except Exception:
                continue

        used_video_ids.add(selected_vid)
        validated_playlist.append(
            {
                "topic_title": f"Module {idx + 1}: {query}",
                "query": query,
                "video_id": selected_vid,
                "embed_url": f"https://www.youtube.com/embed/{selected_vid}",
                "validation_summary": validation_summary,
            }
        )

    if not validated_playlist:
        validated_playlist.append(
            {
                "topic_title": "Module 1: Core Lecture Walkthrough",
                "query": queries[0] if queries else "University Lecture",
                "video_id": "dQw4w9WgXcQ",
                "embed_url": "https://www.youtube.com/embed/dQw4w9WgXcQ",
                "validation_summary": "Fallback video mounted.",
            }
        )

    primary = validated_playlist[0]
    return {
        "video_id": primary["video_id"],
        "embed_url": primary["embed_url"],
        "validation_summary": primary["validation_summary"],
        "validated_criteria": criteria,
        "playlist": validated_playlist,
    }


def _generate_logic_sandbox(extracted_text: str, client: genai.Client) -> Dict[str, Any]:
    prompt = f"""
    Create an Active Practice Sandbox challenge based on these lecture slides.
    The student must pass this challenge to unlock completion.
    Slides:
    {extracted_text[:8000]}
    """
    res = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=LogicSandboxChallenge,
            temperature=0.2,
        ),
    )
    return json.loads(res.text)


# ---------------------------------------------------------------------------
# Path B (SPATIAL): Deep-Dive Spatial Script + HD MP4 Video Compiler
# ---------------------------------------------------------------------------
def _build_deep_spatial_script(extracted_text: str, client: genai.Client) -> Dict[str, str]:
    """
    Splits a multi-slide Spatial lecture (e.g., 40+ slides) into chronological
    batches and generates:
    1. A comprehensive Scene-by-Scene Visual Storyboard covering all slides.
    2. A full-length 2-host studio narration ('Andrew:' and 'Ava:') walking
       the student visually through every diagram, vector, and physical process.
    """
    raw_slides = [
        s.strip() for s in re.split(r"(?=###\s+Slide)", extracted_text) if s.strip()
    ]

    batches: List[str] = []
    if len(raw_slides) > 6:
        slides_per_batch = 12
        for i in range(0, len(raw_slides), slides_per_batch):
            batches.append("\n\n".join(raw_slides[i : i + slides_per_batch]))
    else:
        chunk_size = 10000
        batches = [
            extracted_text[i : i + chunk_size]
            for i in range(0, len(extracted_text), chunk_size)
        ]

    batches = batches[:4]
    total_batches = len(batches)
    compiled_storyboards: List[str] = []
    compiled_narrations: List[str] = []

    for idx, batch_text in enumerate(batches):
        part_num = idx + 1
        print(
            f"[PATH B DEEP-DIVE] Designing Spatial Scenes & Narration Part {part_num}/{total_batches}..."
        )
        prompt = f"""
        You are designing Part {part_num} of {total_batches} of a comprehensive Spatial & Visual Explainer for a university lecture.

        Dissect EVERY slide, diagram, physical mechanism, equation, and spatial relationship in this batch chronologically.
        Return a JSON object with two keys:
        1. "visual_storyboard": A detailed scene-by-scene visual breakdown (3 to 4 scenes for this batch). Format each scene clearly as:
           "Scene X (Slides A-B): [Title] — [Detailed visual description of the geometry, vectors, diagrams, and physical transformations + key equations in LaTeX $...$]"
        2. "spoken_dialogue": A 500-to-700 word conversational walkthrough between `Andrew:` (lead visual instructor) and `Ava:` (co-host) guiding the student's eyes through the diagrams and mechanics on these slides. Every line in spoken_dialogue MUST start with `Andrew:` or `Ava:` with no markdown bolding or bullets.

        SLIDE BATCH ({part_num} of {total_batches}):
        {batch_text}
        """
        try:
            res = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.3,
                ),
            )
            data = json.loads(res.text)
            if data.get("visual_storyboard"):
                sb = data["visual_storyboard"]
                if isinstance(sb, list):
                    sb = "\n\n".join([str(item) for item in sb])
                compiled_storyboards.append(f"### Module {part_num} Visual Storyboard\n{sb}")
            if data.get("spoken_dialogue"):
                dlg = data["spoken_dialogue"]
                if isinstance(dlg, list):
                    dlg = "\n".join([str(item) for item in dlg])
                compiled_narrations.append(dlg)

            if part_num < total_batches:
                time.sleep(2)
        except Exception as exc:
            print(f"[PATH B BATCH WARNING] Part {part_num} failed: {exc}")

    return {
        "storyboard": "\n\n".join(compiled_storyboards),
        "narration_script": "\n\n".join(compiled_narrations),
    }


def _compile_local_spatial_mp4(
    file_hash: str, wav_path: Path, output_mp4_path: Path
) -> bool:
    """
    Renders each PDF slide onto a 1280x720 HD North Star Spatial HUD frame and
    muxes the frames with the studio WAV narration into an H.264 MP4 video.
    """
    try:
        import fitz  # PyMuPDF
        import imageio_ffmpeg
        import subprocess
        import tempfile
        from PIL import Image, ImageDraw
    except ImportError as imp_err:
        print(f"[MP4 COMPILER WARNING] Missing video libraries ({imp_err}).")
        return False

    pdf_path = Path(__file__).resolve().parent / "stored_pdfs" / f"{file_hash}.pdf"
    if not pdf_path.exists() or not wav_path.exists():
        return False

    try:
        with wave.open(str(wav_path), "rb") as wf:
            total_duration = max(wf.getnframes() / float(wf.getframerate()), 5.0)

        doc = fitz.open(str(pdf_path))
        num_pages = len(doc)
        if num_pages == 0:
            return False

        duration_per_slide = total_duration / num_pages

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            frame_Entries: List[str] = []

            for page_idx in range(num_pages):
                page = doc[page_idx]
                pix = page.get_pixmap(dpi=130)
                slide_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Create 1280x720 HD dark-blue North Star canvas
                canvas = Image.new("RGB", (1280, 720), color=(17, 36, 55))
                draw = ImageDraw.Draw(canvas)

                # Top HUD Header Bar
                draw.rectangle([(0, 0), (1280, 48)], fill=(28, 55, 81))
                draw.text(
                    (24, 16),
                    f"NORTH STAR SPATIAL VISUALIZER  |  SLIDE {page_idx + 1} OF {num_pages}",
                    fill=(186, 230, 253),
                )

                # Fit slide cleanly inside the 1240x640 viewport area
                slide_img.thumbnail((1220, 630), Image.Resampling.LANCZOS)
                x_offset = (1280 - slide_img.width) // 2
                y_offset = 54 + (640 - slide_img.height) // 2
                canvas.paste(slide_img, (x_offset, y_offset))

                frame_file = tmp_path / f"slide_{page_idx:04d}.png"
                canvas.save(str(frame_file), "PNG")

                safe_posix_path = str(frame_file).replace("\\", "/")
                frame_Entries.append(f"file '{safe_posix_path}'")
                frame_Entries.append(f"duration {duration_per_slide:.3f}")

            # Repeat final frame for ffmpeg concat demuxer
            last_posix = str(tmp_path / f"slide_{num_pages - 1:04d}.png").replace("\\", "/")
            frame_Entries.append(f"file '{last_posix}'")

            concat_txt = tmp_path / "concat_list.txt"
            concat_txt.write_text("\n".join(frame_Entries), encoding="utf-8")

            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [
                ffmpeg_exe,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_txt),
                "-i",
                str(wav_path),
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-tune",
                "stillimage",
                "-vf",
                "fps=5,format=yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-shortest",
                str(output_mp4_path),
            ]
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )

        doc.close()
        if output_mp4_path.exists() and output_mp4_path.stat().st_size > 1024:
            print(
                f"[PATH B MP4 SUCCESS] Compiled synchronized HD lecture video: {output_mp4_path.name}"
            )
            return True
    except Exception as exc:
        print(f"[MP4 COMPILER WARNING] Could not compile MP4 ({exc}).")

    return False


def _generate_spatial_media(
    file_hash: str,
    extracted_text: str,
    fallback_spatial_prompt: str,
    client: genai.Client,
) -> Dict[str, str]:
    """
    1. Dissects all lecture slides into a deep-dive visual storyboard & studio narration.
    2. Synthesizes 2-host studio audio via Gemini Multi-Speaker TTS.
    3. Compiles a synchronized 720p HD .mp4 video combining the PDF slide deck and studio audio.
    """
    deep_spatial = _build_deep_spatial_script(extracted_text, client)
    full_storyboard = (
        deep_spatial["storyboard"]
        if len(deep_spatial["storyboard"]) > len(fallback_spatial_prompt)
        else fallback_spatial_prompt
    )
    narration_script = (
        deep_spatial["narration_script"]
        if deep_spatial["narration_script"]
        else "Andrew: " + re.sub(r"Scene\s+\d+\s*:", "", fallback_spatial_prompt)
    )

    # Synthesize the full-length studio audio track (.wav or fallback .mp3)
    audio_media_url = _synthesize_audio_resilient(
        text=narration_script,
        file_hash=file_hash,
        suffix="spatial_narration",
        client=client,
    )

    wav_path = MEDIA_DIR / f"{file_hash}_spatial_narration.wav"
    mp4_filename = f"{file_hash}_spatial.mp4"
    mp4_path = MEDIA_DIR / mp4_filename

    # Compile synchronized HD MP4 video from the PDF slides + studio WAV track
    if wav_path.exists() and _compile_local_spatial_mp4(file_hash, wav_path, mp4_path):
        return {
            "media_url": f"/media/{mp4_filename}",
            "storyboard": full_storyboard,
        }

    return {
        "media_url": audio_media_url,
        "storyboard": full_storyboard,
    }


# ---------------------------------------------------------------------------
# Path C (ABSTRACT): Deep-Dive 2-Host Conversational Podcast Synthesis
# ---------------------------------------------------------------------------
def _build_deep_podcast_script(extracted_text: str, client: genai.Client) -> str:
    raw_slides = [
        s.strip() for s in re.split(r"(?=###\s+Slide)", extracted_text) if s.strip()
    ]

    batches: List[str] = []
    if len(raw_slides) > 6:
        slides_per_batch = 15
        for i in range(0, len(raw_slides), slides_per_batch):
            batches.append("\n\n".join(raw_slides[i : i + slides_per_batch]))
    else:
        chunk_size = 10000
        batches = [
            extracted_text[i : i + chunk_size]
            for i in range(0, len(extracted_text), chunk_size)
        ]

    batches = batches[:4]
    total_batches = len(batches)
    compiled_chapters: List[str] = []

    for idx, batch_text in enumerate(batches):
        part_num = idx + 1
        print(
            f"[PATH C DEEP-DIVE] Writing 2-Host podcast script Part {part_num}/{total_batches}..."
        )
        prompt = f"""
        You are writing Part {part_num} of {total_batches} of an engaging, deep-dive educational podcast between two hosts discussing university lecture slides:
        - Andrew: The warm, articulate lead explainer who breaks down complex theories and real-world examples.
        - Ava: The sharp, curious co-host who asks probing questions, points out nuances, and connects concepts together.

        CRITICAL INSTRUCTIONS:
        1. Format EVERY speaker turn strictly starting with `Andrew:` or `Ava:` on a new line (no brackets).
        2. Dissect EVERY slide, definition, theory, and example in this batch chronologically. Do not skip material.
        3. Make the dialogue sound genuinely human—use natural transitions, enthusiasm, analogies, and conversational pacing (500 to 700 words for this part).
        4. Do NOT use markdown bolding, bullet points, or stage directions.
        5. {"Start with a brief, energetic welcome to today's deep dive." if part_num == 1 else "Continue the conversation seamlessly from the previous section without re-introducing the show."}
        6. {"Wrap up with a memorable final takeaway tying the lecture together." if part_num == total_batches else "Transition smoothly into the next group of slides."}

        SLIDE BATCH ({part_num} of {total_batches}):
        {batch_text}
        """
        try:
            res = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.4),
            )
            chapter_script = (res.text or "").strip()
            if chapter_script:
                compiled_chapters.append(chapter_script)
            if part_num < total_batches:
                time.sleep(2)
        except Exception as exc:
            print(f"[PATH C CHAPTER WARNING] Part {part_num} failed: {exc}")

    return "\n\n".join(compiled_chapters)


def _generate_abstract_podcast(
    file_hash: str, extracted_text: str, fallback_script: str, client: genai.Client
) -> Dict[str, str]:
    deep_script = _build_deep_podcast_script(extracted_text, client)
    final_script = (
        deep_script if len(deep_script) > len(fallback_script) else fallback_script
    )

    media_url = _synthesize_audio_resilient(
        text=final_script,
        file_hash=file_hash,
        suffix="podcast",
        client=client,
    )
    return {
        "media_url": media_url,
        "full_script": final_script,
    }


# ---------------------------------------------------------------------------
# Main Phase 3 Dispatcher
# ---------------------------------------------------------------------------
def execute_phase3_generation(
    file_hash: str, extracted_text: str, routing_data: Dict[str, Any]
) -> Dict[str, Any]:
    mark_generating_media(file_hash)
    client = get_genai_client()
    classification = routing_data.get("classification", "LOGIC")

    media_url = ""

    if classification == "LOGIC":
        print(
            f"[PHASE 3 - PATH A] Validating YouTube transcripts & building sandbox for {file_hash}..."
        )
        yt_result = _fetch_and_validate_youtube(
            queries=routing_data.get("youtube_search_queries", []),
            criteria=routing_data.get("transcript_validation_criteria", []),
            client=client,
        )
        sandbox_challenge = _generate_logic_sandbox(extracted_text, client)

        routing_data["youtube_validation"] = yt_result
        routing_data["practice_sandbox"] = sandbox_challenge
        media_url = yt_result["embed_url"]

    elif classification == "SPATIAL":
        print(
            f"[PHASE 3 - PATH B] Generating deep-dive Spatial MP4 video & storyboard for {file_hash}..."
        )
        short_spatial = routing_data.get("spatial_explainer_prompt", "")
        spatial_res = _generate_spatial_media(
            file_hash=file_hash,
            extracted_text=extracted_text,
            fallback_spatial_prompt=short_spatial,
            client=client,
        )
        media_url = spatial_res["media_url"]
        routing_data["spatial_explainer_prompt"] = spatial_res["storyboard"]

    elif classification == "ABSTRACT":
        print(
            f"[PHASE 3 - PATH C] Synthesizing deep-dive Abstract audio podcast for {file_hash}..."
        )
        short_script = routing_data.get("abstract_podcast_script", "")
        podcast_res = _generate_abstract_podcast(
            file_hash=file_hash,
            extracted_text=extracted_text,
            fallback_script=short_script,
            client=client,
        )
        media_url = podcast_res["media_url"]
        routing_data["abstract_podcast_script"] = podcast_res["full_script"]

    save_generated_media(
        file_hash=file_hash,
        media_path=media_url,
        updated_routing_payload=json.dumps(routing_data),
    )
    return routing_data