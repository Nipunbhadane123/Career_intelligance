import os
import bcrypt
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config import SQLALCHEMY_DATABASE_URL
from models import (
    Base, User, Meeting, Participant, ActionItem, KeyPoint, KeyDecision,
    IntegrationSync, AuditLog
)

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Create all tables in SQLite."""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Dependency for obtaining a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ─────────────────────────────────────────────
# USER AUTHENTICATION & ACCESS CONTROL
# ─────────────────────────────────────────────
def create_user(username: str, plain_password: str, role: str = "member") -> Optional[Dict[str, Any]]:
    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.username == username.strip()).first()
        if existing:
            return None
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(plain_password.encode('utf-8'), salt).decode('utf-8')
        user = User(username=username.strip(), password_hash=hashed, role=role)
        db.add(user)
        db.commit()
        db.refresh(user)
        return {"id": user.id, "username": user.username, "role": user.role}
    finally:
        db.close()

def authenticate_user(username: str, plain_password: str) -> Optional[Dict[str, Any]]:
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username.strip()).first()
        if not user:
            return None
        if bcrypt.checkpw(plain_password.encode('utf-8'), user.password_hash.encode('utf-8')):
            return {"id": user.id, "username": user.username, "role": user.role}
        return None
    finally:
        db.close()

def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return None
        return {"id": user.id, "username": user.username, "role": user.role, "created_at": user.created_at}
    finally:
        db.close()

def can_user_access_meeting(user_id: Optional[int], meeting_id: int) -> bool:
    """
    Task 7: Strict User Access Control.
    Returns True if:
    - user is admin
    - meeting has no user_id (public demo meeting)
    - meeting user_id matches user_id
    Returns False otherwise.
    """
    if user_id is None:
        return False
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if user and user.role == "admin":
            return True
        meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
        if not meeting:
            return False
        # Public seed meetings without owner are accessible, or meeting owned by user
        if meeting.user_id is None or meeting.user_id == user_id:
            return True
        return False
    finally:
        db.close()

# ─────────────────────────────────────────────
# MEETING CRUD OPERATIONS
# ─────────────────────────────────────────────
def get_user_meetings(
    user_id: Optional[int] = None,
    platform: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Retrieve meetings accessible to a specific user, with optional filters."""
    db = SessionLocal()
    try:
        query = db.query(Meeting)
        if user_id is not None:
            # Show meetings owned by user or shared/public meetings
            query = query.filter((Meeting.user_id == user_id) | (Meeting.user_id.is_(None)))
        if platform:
            query = query.filter(Meeting.platform == platform)
        if start_date:
            if isinstance(start_date, str):
                start_date = datetime.fromisoformat(start_date)
            query = query.filter(Meeting.created_at >= start_date)
        if end_date:
            if isinstance(end_date, str):
                end_date = datetime.fromisoformat(end_date)
            query = query.filter(Meeting.created_at <= end_date)
            
        meetings = query.order_by(Meeting.created_at.desc()).all()
        return [format_meeting_dict(m) for m in meetings]
    finally:
        db.close()

def get_all_meetings(start_date=None, end_date=None) -> List[Dict[str, Any]]:
    """Retrieve all meetings in repository."""
    db = SessionLocal()
    try:
        query = db.query(Meeting)
        if start_date:
            if isinstance(start_date, str):
                start_date = datetime.fromisoformat(start_date)
            query = query.filter(Meeting.created_at >= start_date)
        if end_date:
            if isinstance(end_date, str):
                end_date = datetime.fromisoformat(end_date)
            query = query.filter(Meeting.created_at <= end_date)
        meetings = query.order_by(Meeting.id.asc()).all()
        return [format_meeting_dict(m) for m in meetings]
    finally:
        db.close()

def get_meeting_by_id(meeting_id: int, user_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Retrieve single meeting with access verification."""
    db = SessionLocal()
    try:
        m = db.query(Meeting).filter(Meeting.id == meeting_id).first()
        if not m:
            return None
        # If user_id provided, enforce access control
        if user_id is not None and m.user_id is not None and m.user_id != user_id:
            user = db.query(User).filter(User.id == user_id).first()
            if not (user and user.role == "admin"):
                return None  # Access denied
        return format_meeting_dict(m)
    finally:
        db.close()

def save_meeting_to_db(
    filename: str,
    transcript: str,
    summary: str,
    action_items_data: List[Dict[str, Any]],
    key_points_data: List[str],
    key_decisions_data: List[str],
    participants_data: List[str],
    duration_seconds: float = 0.0,
    platform: str = "upload",
    external_id: Optional[str] = None,
    user_id: Optional[int] = None,
    title: Optional[str] = None,
    **kwargs
) -> int:
    """Save newly processed meeting and all relational entities to database."""
    db = SessionLocal()
    try:
        clean_title = (title or "").strip() or filename.rsplit('.', 1)[0].replace('_', ' ').title()
        meeting = Meeting(
            filename=filename,
            title=clean_title,
            transcript=transcript,
            summary=summary,
            duration_seconds=duration_seconds,
            platform=platform,
            external_id=external_id,
            user_id=user_id,
            created_at=datetime.utcnow()
        )
        db.add(meeting)
        db.flush()

        # Link or create participants
        participant_objs = {}
        for p_name in participants_data:
            clean_name = p_name.strip()
            if not clean_name:
                continue
            part = db.query(Participant).filter(Participant.name == clean_name).first()
            if not part:
                part = Participant(name=clean_name)
                db.add(part)
                db.flush()
            participant_objs[clean_name] = part
            if part not in meeting.participants:
                meeting.participants.append(part)

        # Key Points
        for kp_text in key_points_data:
            if kp_text.strip():
                db.add(KeyPoint(meeting_id=meeting.id, point=kp_text.strip()))

        # Key Decisions
        for kd_text in key_decisions_data:
            if kd_text.strip():
                db.add(KeyDecision(meeting_id=meeting.id, decision=kd_text.strip()))

        # Action Items
        for ai in action_items_data:
            desc = ai.get("description", "").strip()
            if not desc:
                continue
            assignee_name = ai.get("assigned_participant", "Unassigned")
            part_id = participant_objs[assignee_name].id if assignee_name in participant_objs else None
            db.add(ActionItem(
                meeting_id=meeting.id,
                participant_id=part_id,
                description=desc,
                deadline=ai.get("deadline"),
                priority=ai.get("priority", "Medium"),
                status=ai.get("status", "Pending")
            ))

        db.commit()
        db.refresh(meeting)
        return meeting.id
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

def delete_meeting(meeting_id: int, user_id: Optional[int] = None) -> bool:
    """Delete a meeting if authorized."""
    db = SessionLocal()
    try:
        meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
        if not meeting:
            return False
        if user_id is not None and meeting.user_id is not None and meeting.user_id != user_id:
            user = db.query(User).filter(User.id == user_id).first()
            if not (user and user.role == "admin"):
                return False
        db.delete(meeting)
        db.commit()
        return True
    finally:
        db.close()

def format_meeting_dict(m: Meeting) -> Dict[str, Any]:
    return {
        "id": m.id,
        "filename": m.filename,
        "title": m.title or m.filename.rsplit('.', 1)[0].replace('_', ' ').title(),
        "transcript": m.transcript,
        "summary": m.summary,
        "duration_seconds": m.duration_seconds or 0.0,
        "platform": m.platform or "upload",
        "external_id": m.external_id,
        "created_at": m.created_at,
        "user_id": m.user_id,
        "action_items": [
            {
                "id": a.id,
                "description": a.description,
                "assigned_participant": a.participant.name if a.participant else "Unassigned",
                "deadline": a.deadline,
                "priority": a.priority or "Medium",
                "status": a.status or "Pending"
            }
            for a in (m.action_items or [])
        ],
        "key_points": [kp.point for kp in (m.key_points or [])],
        "key_decisions": [kd.decision for kd in (m.key_decisions or [])],
        "participants": [p.name for p in (m.participants or [])]
    }

# ─────────────────────────────────────────────
# INTEGRATION SYNC & DUPLICATE TRACKING
# ─────────────────────────────────────────────
def compute_file_hash(content: bytes) -> str:
    """Compute SHA-256 hash of file content for duplicate recording detection."""
    return hashlib.sha256(content).hexdigest()

def is_duplicate_sync(platform: str, external_id: Optional[str] = None, file_hash: Optional[str] = None) -> bool:
    """
    Task 4 & 5: Check if recording has already been imported/processed.
    Prevents duplicate recordings by external UUID or content hash.
    """
    db = SessionLocal()
    try:
        query = db.query(IntegrationSync).filter(
            IntegrationSync.platform == platform,
            IntegrationSync.status == "completed"
        )
        if external_id:
            match = query.filter(IntegrationSync.external_id == external_id).first()
            if match:
                return True
        if file_hash:
            match = query.filter(IntegrationSync.file_hash == file_hash).first()
            if match:
                return True
        return False
    finally:
        db.close()

def record_sync_attempt(
    platform: str,
    external_id: str,
    file_name: str,
    file_hash: Optional[str] = None,
    status: str = "pending"
) -> int:
    """Record sync attempt for auditing and duplicate prevention."""
    db = SessionLocal()
    try:
        sync = IntegrationSync(
            platform=platform,
            external_id=external_id,
            file_name=file_name,
            file_hash=file_hash,
            status=status,
            synced_at=datetime.utcnow()
        )
        db.add(sync)
        db.commit()
        db.refresh(sync)
        return sync.id
    finally:
        db.close()

def update_sync_record(
    sync_id: int,
    status: str,
    meeting_id: Optional[int] = None,
    error_message: Optional[str] = None
):
    """Update status of integration sync job."""
    db = SessionLocal()
    try:
        sync = db.query(IntegrationSync).filter(IntegrationSync.id == sync_id).first()
        if sync:
            sync.status = status
            if meeting_id:
                sync.meeting_id = meeting_id
            if error_message:
                sync.error_message = error_message
            db.commit()
    finally:
        db.close()

def get_sync_history(platform: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve recent integration sync jobs."""
    db = SessionLocal()
    try:
        query = db.query(IntegrationSync)
        if platform:
            query = query.filter(IntegrationSync.platform == platform)
        syncs = query.order_by(IntegrationSync.synced_at.desc()).limit(limit).all()
        return [
            {
                "id": s.id,
                "platform": s.platform,
                "external_id": s.external_id,
                "file_name": s.file_name,
                "status": s.status,
                "meeting_id": s.meeting_id,
                "error_message": s.error_message,
                "synced_at": str(s.synced_at)
            }
            for s in syncs
        ]
    finally:
        db.close()

# ─────────────────────────────────────────────
# AUDIT LOGGING
# ─────────────────────────────────────────────
def log_audit(user_id: Optional[int], action: str, resource: str, status: str = "SUCCESS"):
    db = SessionLocal()
    try:
        log = AuditLog(user_id=user_id, action=action, resource=resource, status=status)
        db.add(log)
        db.commit()
    finally:
        db.close()

def export_meeting_knowledge_units(meeting_dict: dict) -> List[Dict[str, Any]]:
    """
    Organizes meeting information into discrete structured units for AI search:
    - Metadata
    - Summary
    - Key Decisions
    - Action Items (with deadlines & assignees)
    - Transcript Chunks
    """
    if not meeting_dict or "id" not in meeting_dict:
        return []
    m_id = meeting_dict["id"]
    filename = meeting_dict.get("filename") or f"Meeting_{m_id}"
    created_at_str = str(meeting_dict.get("created_at") or "")
    user_id = meeting_dict.get("user_id")
    platform = meeting_dict.get("platform") or "upload"
    
    units = []

    # 1. Summary Unit
    if meeting_dict.get("summary"):
        units.append({
            "unit_id": f"m_{m_id}_summary",
            "meeting_id": m_id,
            "meeting_filename": filename,
            "doc_type": "summary",
            "title": f"Executive Summary - {filename}",
            "text": f"Meeting: {filename}\nExecutive Summary: {meeting_dict['summary']}",
            "metadata": {
                "meeting_id": m_id,
                "meeting_filename": filename,
                "doc_type": "summary",
                "created_at": created_at_str,
                "user_id": user_id if user_id is not None else -1,
                "platform": platform
            }
        })

    # 2. Decisions Units
    for idx, dec in enumerate(meeting_dict.get("key_decisions") or []):
        if not dec:
            continue
        units.append({
            "unit_id": f"m_{m_id}_decision_{idx}",
            "meeting_id": m_id,
            "meeting_filename": filename,
            "doc_type": "decision",
            "title": f"Decision in {filename}",
            "text": f"Meeting: {filename}\nKey Decision: {dec}",
            "metadata": {
                "meeting_id": m_id,
                "meeting_filename": filename,
                "doc_type": "decision",
                "decision_idx": idx,
                "created_at": created_at_str,
                "user_id": user_id if user_id is not None else -1,
                "platform": platform
            }
        })

    # 3. Action Items Units
    for idx, ai in enumerate(meeting_dict.get("action_items") or []):
        if not isinstance(ai, dict):
            continue
        desc = ai.get("description", "")
        assignee = ai.get("assigned_participant") or "Unassigned"
        deadline = ai.get("deadline") or "Not specified"
        priority = ai.get("priority") or "Normal"
        status = ai.get("status") or "Pending"
        
        text_repr = (
            f"Meeting: {filename}\n"
            f"Action Item: {desc}\n"
            f"Assigned To: {assignee}\n"
            f"Deadline: {deadline}\n"
            f"Priority: {priority}\n"
            f"Status: {status}"
        )
        units.append({
            "unit_id": f"m_{m_id}_action_{idx}",
            "meeting_id": m_id,
            "meeting_filename": filename,
            "doc_type": "action_item",
            "title": f"Action Item in {filename}",
            "text": text_repr,
            "metadata": {
                "meeting_id": m_id,
                "meeting_filename": filename,
                "doc_type": "action_item",
                "action_id": ai.get("id", idx),
                "assignee": assignee,
                "deadline": deadline,
                "priority": priority,
                "status": status,
                "created_at": created_at_str,
                "user_id": user_id if user_id is not None else -1,
                "platform": platform
            }
        })

    # 4. Transcript Chunks Units
    transcript = meeting_dict.get("transcript") or ""
    lines = [line.strip() for line in transcript.split("\n") if line.strip()]
    chunk_size = 4
    for i in range(0, len(lines), chunk_size):
        chunk_lines = lines[i:i + chunk_size]
        chunk_text = "\n".join(chunk_lines)
        chunk_idx = i // chunk_size
        units.append({
            "unit_id": f"m_{m_id}_chunk_{chunk_idx}",
            "meeting_id": m_id,
            "meeting_filename": filename,
            "doc_type": "transcript",
            "title": f"Transcript Section in {filename}",
            "text": f"Meeting: {filename} (Transcript Section):\n{chunk_text}",
            "metadata": {
                "meeting_id": m_id,
                "meeting_filename": filename,
                "doc_type": "transcript",
                "chunk_idx": chunk_idx,
                "created_at": created_at_str,
                "user_id": user_id if user_id is not None else -1,
                "platform": platform
            }
        })

    return units

# ─────────────────────────────────────────────
# SEED & REPOSITORY VERIFICATION
# ─────────────────────────────────────────────
def verify_knowledge_repository() -> Dict[str, Any]:
    db = SessionLocal()
    try:
        total_meetings = db.query(Meeting).count()
        total_actions = db.query(ActionItem).count()
        total_decisions = db.query(KeyDecision).count()
        participants = [p.name for p in db.query(Participant).all()]
        return {
            "status": "PASS" if total_meetings > 0 else "EMPTY",
            "total_meetings": total_meetings,
            "total_action_items": total_actions,
            "total_decisions": total_decisions,
            "total_participants": participants
        }
    finally:
        db.close()

