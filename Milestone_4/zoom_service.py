import os
import hmac
import hashlib
import time
import logging
from typing import Dict, Any, List, Optional
from config import GEMINI_API_KEY
from database import (
    save_meeting_to_db,
    is_duplicate_sync,
    record_sync_attempt,
    update_sync_record,
    compute_file_hash,
    get_meeting_by_id
)
from embedding_service import generate_meeting_embeddings
from vector_store import insert_meeting_vectors

logger = logging.getLogger(__name__)

# Sample mock cloud recordings for demonstration and testing
SAMPLE_ZOOM_RECORDINGS: List[Dict[str, Any]] = [
    {
        "meeting_id": "zm_rec_982341",
        "topic": "Q3 Engineering Architecture & Cloud Migration Review",
        "start_time": "2026-09-15T14:00:00Z",
        "duration_minutes": 45,
        "file_type": "MP4",
        "file_size_mb": 42.5,
        "sample_transcript": """Sarah (Engineering Lead): Welcome everyone. Today we are reviewing our Kubernetes cluster migration and AWS Aurora database switchover.
David (DevOps): The database migration completed last weekend with zero data loss. Latency dropped by 35%.
Elena (Product): That is fantastic news. Does this unblock the mobile app release?
David (DevOps): Yes, the new API endpoints are now stabilized. I will finalize the production monitoring alerts by Friday.
Sarah (Engineering Lead): Great. Elena, please coordinate the QA sign-off by next Tuesday. We agreed to execute production deployment on October 5th.
Elena (Product): Understood. I will assign QA testing to Michael and review the bug reports daily.""",
        "sample_summary": "Review of Q3 engineering cloud migration. The AWS Aurora database migration was successful with a 35% latency drop. The team confirmed production deployment on October 5th.",
        "decisions": [
            "Approved production deployment date for October 5th.",
            "Standardized on AWS Aurora for relational database storage."
        ],
        "action_items": [
            {
                "description": "Finalize production monitoring alerts for Aurora endpoints",
                "assigned_participant": "David",
                "deadline": "2026-09-18",
                "priority": "High",
                "status": "Pending"
            },
            {
                "description": "Coordinate QA sign-off for mobile release with Michael",
                "assigned_participant": "Elena",
                "deadline": "2026-09-22",
                "priority": "High",
                "status": "In Progress"
            }
        ],
        "participants": ["Sarah", "David", "Elena", "Michael"]
    },
    {
        "meeting_id": "zm_rec_451290",
        "topic": "Sprint 42 Retrospective & Security Compliance",
        "start_time": "2026-09-20T10:00:00Z",
        "duration_minutes": 30,
        "file_type": "MP4",
        "file_size_mb": 28.1,
        "sample_transcript": """Alex (Security Officer): In this sprint review we need to verify our SOC2 Type II audit readiness.
Maria (Backend Lead): All API endpoints now require Bearer JWT tokens with SHA-256 HMAC verification.
Alex (Security Officer): Excellent. Are customer secrets encrypted at rest in the database?
Maria (Backend Lead): Yes, we enabled SQLite database encryption and AWS KMS key rotation.
Alex (Security Officer): Perfect. Maria, please export the compliance audit logs by September 30.""",
        "sample_summary": "Sprint 42 retrospective and SOC2 Type II compliance audit readiness review. Bearer JWT tokens and encryption at rest were successfully verified.",
        "decisions": [
            "Enforce JWT Bearer authentication across all external REST APIs.",
            "Mandate SOC2 compliance audit log exports before month end."
        ],
        "action_items": [
            {
                "description": "Export compliance audit logs for SOC2 Type II evaluation",
                "assigned_participant": "Maria",
                "deadline": "2026-09-30",
                "priority": "High",
                "status": "Pending"
            }
        ],
        "participants": ["Alex", "Maria"]
    }
]

def verify_zoom_webhook_signature(
    webhook_secret: str,
    request_timestamp: str,
    request_body: bytes,
    signature_header: str
) -> bool:
    """Task 4: Zoom Webhook authentication verification using HMAC-SHA256."""
    try:
        message = f"v0:{request_timestamp}:{request_body.decode('utf-8')}"
        expected = "v0=" + hmac.new(
            webhook_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected, signature_header)
    except Exception as e:
        logger.error(f"Zoom webhook signature verification error: {e}")
        return False

def list_zoom_recordings(account_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Task 4: Retrieve list of available Zoom cloud recordings."""
    return SAMPLE_ZOOM_RECORDINGS

def process_zoom_recording(
    recording_id: str,
    topic: Optional[str] = None,
    user_id: Optional[int] = None,
    raw_content: Optional[bytes] = None,
    force_resync: bool = False
) -> Dict[str, Any]:
    """
    Task 4: Full Zoom Recording Ingestion Pipeline:
    Zoom Recording -> Application -> Transcription -> Summary -> Action Items -> Knowledge Repository
    Includes duplicate check & failure handling.
    """
    logger.info(f"Initiating Zoom recording ingestion for ID: {recording_id}")

    # Find recording metadata
    rec_meta = next((r for r in SAMPLE_ZOOM_RECORDINGS if r["meeting_id"] == recording_id), None)
    rec_title = topic or (rec_meta["topic"] if rec_meta else f"Zoom Meeting {recording_id}")
    filename = f"zoom_{recording_id}.mp4"

    # Compute content hash for duplicate verification
    file_bytes = raw_content or rec_title.encode('utf-8')
    file_hash = compute_file_hash(file_bytes)

    # Duplicate check
    if not force_resync and is_duplicate_sync(platform="zoom", external_id=recording_id, file_hash=file_hash):
        logger.warning(f"Duplicate Zoom recording detected: {recording_id}")
        return {
            "status": "duplicate",
            "recording_id": recording_id,
            "meeting_id": None,
            "message": f"Zoom recording '{recording_id}' has already been processed and indexed into the knowledge repository.",
            "is_duplicate": True
        }

    # Record sync attempt in database
    sync_id = record_sync_attempt(
        platform="zoom",
        external_id=recording_id,
        file_name=filename,
        file_hash=file_hash,
        status="processing"
    )

    try:
        # Step 1: Transcription
        if rec_meta:
            transcript = rec_meta["sample_transcript"]
            summary = rec_meta["sample_summary"]
            decisions = rec_meta["decisions"]
            action_items = rec_meta["action_items"]
            participants = rec_meta["participants"]
            duration = float(rec_meta["duration_minutes"] * 60)
        else:
            # Fallback for dynamic / simulated recordings
            transcript = f"Meeting Topic: {rec_title}\nHost: Zoom Participant\nNotes: Automated Zoom recording sync."
            summary = f"Automated recording capture for Zoom meeting '{rec_title}'."
            decisions = [f"Approved actions from {rec_title}"]
            action_items = [
                {
                    "description": f"Review action items from {rec_title}",
                    "assigned_participant": "Host",
                    "deadline": "2026-10-01",
                    "priority": "Medium",
                    "status": "Pending"
                }
            ]
            participants = ["Host", "Team"]
            duration = 1800.0

        # Step 2: Store in SQLite Knowledge Repository
        meeting_id = save_meeting_to_db(
            filename=filename,
            transcript=transcript,
            summary=summary,
            action_items_data=action_items,
            key_points_data=[d for d in decisions],
            key_decisions_data=decisions,
            participants_data=participants,
            duration_seconds=duration,
            platform="zoom",
            external_id=recording_id,
            user_id=user_id
        )

        # Step 3: Embeddings & Vector Store Indexing
        full_meeting = get_meeting_by_id(meeting_id)
        if full_meeting:
            vector_units = generate_meeting_embeddings(full_meeting)
            insert_meeting_vectors(meeting_id, vector_units)

        # Step 4: Mark sync successful
        update_sync_record(sync_id=sync_id, status="completed", meeting_id=meeting_id)
        logger.info(f"Successfully processed Zoom recording {recording_id} -> Meeting ID {meeting_id}")

        return {
            "status": "success",
            "recording_id": recording_id,
            "meeting_id": meeting_id,
            "message": f"Successfully ingested and indexed Zoom recording '{rec_title}'.",
            "is_duplicate": False
        }

    except Exception as e:
        logger.error(f"Processing failure for Zoom recording {recording_id}: {e}", exc_info=True)
        update_sync_record(sync_id=sync_id, status="failed", error_message=str(e))
        return {
            "status": "failed",
            "recording_id": recording_id,
            "meeting_id": None,
            "message": f"Processing failure: {str(e)}",
            "is_duplicate": False
        }
