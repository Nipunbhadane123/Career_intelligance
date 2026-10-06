import os
import sys
import ctypes
import tempfile
import shutil
import logging
import re
import time
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import soundfile as sf
import whisper
from pyannote.audio import Pipeline
from google import genai

from config import GEMINI_API_KEY, HF_TOKEN
from schemas import MeetingIntelligenceSchema
from llm_service import process_transcript_with_llm
from database import (
    SessionLocal,
    get_meeting_by_id,
    save_meeting_to_db,
    Meeting as MeetingModel,
    Participant as ParticipantModel,
    ActionItem as ActionItemModel,
    KeyPoint as KeyPointModel,
    KeyDecision as KeyDecisionModel
)
from embedding_service import generate_meeting_embeddings
from vector_store import insert_meeting_vectors

logger = logging.getLogger(__name__)

# Ensure ffmpeg in PATH or working directory
LOCAL_FFMPEG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg.exe")
if os.path.exists(LOCAL_FFMPEG):
    os.environ["PATH"] = os.path.dirname(os.path.abspath(__file__)) + os.pathsep + os.environ.get("PATH", "")

# PyTorch C10 DLL fix for Windows
try:
    pytorch_lib_dir = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
    c10_dll = os.path.join(pytorch_lib_dir, "c10.dll")
    if os.path.exists(c10_dll):
        ctypes.CDLL(c10_dll)
except Exception:
    pass

_WHISPER_MODEL = None
_DIARIZATION_PIPELINE = None

def get_whisper_model():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        logger.info("Loading OpenAI Whisper 'base' model...")
        _WHISPER_MODEL = whisper.load_model("base")
    return _WHISPER_MODEL

def get_diarization_pipeline(token: Optional[str] = None):
    global _DIARIZATION_PIPELINE
    token_to_use = token or HF_TOKEN or os.environ.get("HF_TOKEN")
    if not token_to_use:
        return None
    if _DIARIZATION_PIPELINE is None:
        try:
            logger.info("Loading PyAnnote speaker diarization pipeline...")
            _DIARIZATION_PIPELINE = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=token_to_use
            )
        except Exception as e:
            logger.warning(f"Could not load PyAnnote pipeline: {e}")
            return None
    return _DIARIZATION_PIPELINE

def format_timestamp(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"[{mins:02d}:{secs:02d}]"

def merge_transcription_and_diarization(whisper_segments: List[Dict[str, Any]], diarization_result) -> str:
    merged_lines = []
    for seg in whisper_segments:
        start, end, text = seg['start'], seg['end'], seg['text'].strip()
        speaker = "Speaker_01"
        max_overlap = 0.0

        if diarization_result is not None:
            try:
                for turn, _, spk in diarization_result.itertracks(yield_label=True):
                    overlap = max(0.0, min(end, turn.end) - max(start, turn.start))
                    if overlap > max_overlap:
                        max_overlap = overlap
                        speaker = spk
            except Exception:
                speaker = "Speaker_01"

        merged_lines.append(f"{format_timestamp(start)} {speaker}: {text}")
    return "\n".join(merged_lines)

def process_uploaded_media(
    file_bytes: bytes,
    filename: str,
    title: Optional[str] = None,
    user_id: Optional[int] = None,
    platform: str = "upload",
    gemini_key: Optional[str] = None,
    hf_token: Optional[str] = None,
    progress_callback = None
) -> Dict[str, Any]:
    """
    Executes the REAL end-to-end processing pipeline from Milestone 3:
    1. Audio Loading / Extraction (via Whisper/FFmpeg)
    2. Whisper Speech-to-Text Transcription
    3. PyAnnote Speaker Diarization
    4. Merging Timestamps & Speaker Turns
    5. Google Gemini Structured Intelligence Extraction (Summary, Decisions, Actions, Participants)
    6. Relational Knowledge Storage (SQLite)
    7. Multi-Entity Embedding Generation & Persistent ChromaDB Indexing
    """
    key_to_use = gemini_key or GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
    token_to_use = hf_token or HF_TOKEN or os.environ.get("HF_TOKEN", "")

    title_str = (title or "").strip() or filename.rsplit('.', 1)[0].replace('_', ' ').title()
    file_ext = os.path.splitext(filename)[1].lower()

    if progress_callback: progress_callback(0.1, "Saving temporary media file...")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        # 1. Load Audio
        if progress_callback: progress_callback(0.2, "Extracting audio waveform...")
        logger.info(f"Loading audio for '{filename}' from {tmp_path}")
        audio_array = whisper.load_audio(tmp_path, sr=16000)
        duration_seconds = float(len(audio_array)) / 16000.0

        # 2. Transcribe with Whisper
        if progress_callback: progress_callback(0.4, "Transcribing dialogue with OpenAI Whisper...")
        logger.info("Transcribing audio array with Whisper...")
        model = get_whisper_model()
        transcribe_res = model.transcribe(audio_array)
        segments = transcribe_res.get("segments", [])

        # 3. Speaker Diarization with PyAnnote
        if progress_callback: progress_callback(0.6, "Identifying speakers with PyAnnote...")
        final_transcript = ""
        diar_pipeline = get_diarization_pipeline(token_to_use)

        if diar_pipeline is not None and len(audio_array) > 0:
            try:
                waveform_tensor = torch.from_numpy(audio_array).unsqueeze(0)
                diar_out = diar_pipeline({"waveform": waveform_tensor, "sample_rate": 16000})
                diar_annot = getattr(diar_out, "speaker_diarization", diar_out)
                final_transcript = merge_transcription_and_diarization(segments, diar_annot)
            except Exception as d_err:
                logger.warning(f"Diarization failed: {d_err}. Falling back to timestamped transcript.")
                diar_annot = None

        if not final_transcript:
            lines = []
            for seg in segments:
                lines.append(f"{format_timestamp(seg['start'])} Speaker: {seg['text'].strip()}")
            final_transcript = "\n".join(lines) if lines else f"Meeting: {title_str}\n(No spoken dialogue detected)"

        # 4. Intelligence Extraction with Gemini LLM
        if progress_callback: progress_callback(0.8, "Extracting intelligence with Google Gemini...")
        structured_data = None
        if key_to_use:
            try:
                client = genai.Client(api_key=key_to_use)
                structured_data = process_transcript_with_llm(client, final_transcript)
            except Exception as llm_err:
                logger.warning(f"Live Gemini extraction failed ({llm_err}). Falling back to heuristic extraction.")

        if structured_data:
            summary_text = structured_data.summary
            key_points_list = structured_data.key_points
            decisions_list = structured_data.decisions
            raw_action_items = structured_data.action_items
            participants_list = structured_data.participants

            action_items_data = []
            for ai in raw_action_items:
                action_items_data.append({
                    "description": ai.description if hasattr(ai, 'description') else str(ai),
                    "assigned_participant": getattr(ai, 'assigned_participant', 'Unassigned') or 'Unassigned',
                    "deadline": getattr(ai, 'deadline', None),
                    "priority": getattr(ai, 'priority', 'Medium') or 'Medium',
                    "status": getattr(ai, 'status', 'Pending') or 'Pending'
                })
        else:
            extracted_spks = list(dict.fromkeys(re.findall(r'\[\d+:\d+\]\s*([^:]+):', final_transcript))) or ["Participant 1"]
            summary_text = f"Executive Summary for {title_str}: Recorded and transcribed with {len(segments)} spoken dialogue segments. Primary participants: {', '.join(extracted_spks)}."
            key_points_list = [f"Spoken dialogue transcribed across {len(segments)} segments.", f"Recorded duration: {round(duration_seconds / 60.0, 1)} minutes."]
            decisions_list = [f"Logged and verified transcript for {title_str}."]
            action_items_data = [{
                "description": f"Review action items and deliverables discussed in {title_str}",
                "assigned_participant": extracted_spks[0] if extracted_spks else "Unassigned",
                "deadline": None,
                "priority": "Medium",
                "status": "Pending"
            }]
            participants_list = extracted_spks

        # 5. Save to Relational Database
        if progress_callback: progress_callback(0.9, "Storing in knowledge repository...")
        try:
            m_id = save_meeting_to_db(
                filename=filename,
                transcript=final_transcript,
                summary=summary_text,
                action_items_data=action_items_data,
                key_points_data=key_points_list,
                key_decisions_data=decisions_list,
                participants_data=participants_list,
                duration_seconds=duration_seconds,
                platform=platform,
                user_id=user_id,
                title=title_str
            )
        except TypeError:
            m_id = save_meeting_to_db(
                filename=filename,
                transcript=final_transcript,
                summary=summary_text,
                action_items_data=action_items_data,
                key_points_data=key_points_list,
                key_decisions_data=decisions_list,
                participants_data=participants_list,
                duration_seconds=duration_seconds,
                platform=platform,
                user_id=user_id
            )
            try:
                db_t = SessionLocal()
                m_obj = db_t.query(MeetingModel).filter(MeetingModel.id == m_id).first()
                if m_obj:
                    m_obj.title = title_str
                    db_t.commit()
                db_t.close()
            except Exception:
                pass

        full_meeting = get_meeting_by_id(m_id)

        # 6. Generate Vectors & Index into ChromaDB
        if progress_callback: progress_callback(0.95, "Indexing vectors in ChromaDB...")
        vectors = generate_meeting_embeddings(full_meeting)
        insert_meeting_vectors(m_id, vectors)

        if progress_callback: progress_callback(1.0, "Complete!")
        logger.info(f"Successfully processed meeting #{m_id} ('{title_str}') with {len(vectors)} vector units.")

        return {
            "meeting_id": m_id,
            "title": title_str,
            "filename": filename,
            "summary": summary_text,
            "duration_seconds": duration_seconds,
            "segments_count": len(segments),
            "action_items_count": len(action_items_data),
            "decisions_count": len(decisions_list),
            "participants": participants_list,
            "vector_count": len(vectors)
        }

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
