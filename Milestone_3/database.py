import os
import bcrypt
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# SQLite database file located in the Milestone_3 directory
DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meeting_intelligence.db")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_FILE}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def init_db():
    import models
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_user(username: str, plain_password: str) -> bool:
    from models import User
    db = SessionLocal()
    try:
        if db.query(User).filter(User.username == username).first():
            return False
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(plain_password.encode('utf-8'), salt)
        user = User(username=username, password_hash=hashed.decode('utf-8'))
        db.add(user)
        db.commit()
        return True
    finally:
        db.close()

def authenticate_user(username: str, plain_password: str):
    from models import User
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        if not user:
            return None
        if bcrypt.checkpw(plain_password.encode('utf-8'), user.password_hash.encode('utf-8')):
            return {"id": user.id, "username": user.username}
        return None
    finally:
        db.close()

def get_user_meetings(user_id: int = None, start_date = None, end_date = None):
    """Retrieve historical meetings list ordered by most recent with optional date range filter."""
    from models import Meeting
    db = SessionLocal()
    try:
        query = db.query(Meeting)
        if user_id:
            query = query.filter(Meeting.user_id == user_id)
        if start_date:
            if isinstance(start_date, str):
                start_date = datetime.fromisoformat(start_date)
            query = query.filter(Meeting.created_at >= start_date)
        if end_date:
            if isinstance(end_date, str):
                end_date = datetime.fromisoformat(end_date)
            query = query.filter(Meeting.created_at <= end_date)
        meetings = query.order_by(Meeting.created_at.desc()).all()
        return [(m.id, m.filename, m.transcript, m.summary, m.created_at) for m in meetings]
    finally:
        db.close()

def get_meeting_by_id(meeting_id: int):
    """Retrieve complete relational graph for a single meeting."""
    from models import Meeting
    db = SessionLocal()
    try:
        m = db.query(Meeting).filter(Meeting.id == meeting_id).first()
        if not m:
            return None
        return {
            "id": m.id,
            "filename": m.filename,
            "transcript": m.transcript,
            "summary": m.summary,
            "created_at": m.created_at,
            "user_id": m.user_id,
            "action_items": [
                {
                    "id": a.id,
                    "description": a.description,
                    "assigned_participant": a.participant.name if a.participant else "Unassigned",
                    "deadline": a.deadline,
                    "priority": a.priority,
                    "status": a.status
                }
                for a in (m.action_items or [])
            ],
            "key_points": [kp.point for kp in (m.key_points or [])],
            "key_decisions": [kd.decision for kd in (m.key_decisions or [])],
            "participants": [p.name for p in (m.participants or [])]
        }
    finally:
        db.close()

def get_all_meetings(start_date = None, end_date = None):
    """Retrieve all meetings with full relational data for knowledge repository indexing."""
    from models import Meeting
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
        result = []
        for m in meetings:
            result.append({
                "id": m.id,
                "filename": m.filename,
                "transcript": m.transcript,
                "summary": m.summary,
                "created_at": m.created_at,
                "user_id": m.user_id,
                "action_items": [
                    {
                        "id": a.id,
                        "description": a.description,
                        "assigned_participant": a.participant.name if a.participant else "Unassigned",
                        "deadline": a.deadline,
                        "priority": a.priority,
                        "status": a.status
                    }
                    for a in (m.action_items or [])
                ],
                "key_points": [kp.point for kp in (m.key_points or [])],
                "key_decisions": [kd.decision for kd in (m.key_decisions or [])],
                "participants": [p.name for p in (m.participants or [])]
            })
        return result
    finally:
        db.close()

def export_meeting_knowledge_units(meeting_dict: dict):
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
                "created_at": created_at_str
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
                "created_at": created_at_str
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
                "created_at": created_at_str
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
                "created_at": created_at_str
            }
        })

    return units

def verify_knowledge_repository():
    """
    Task 1 Verification:
    Verify that existing database records can be retrieved correctly and linked to the corresponding meeting.
    Checks:
    - Meetings count and metadata
    - Transcripts and summaries present
    - Linked decisions
    - Linked action items and deadlines
    - Linked participants
    """
    meetings = get_all_meetings()
    report = {
        "status": "PASS",
        "total_meetings": len(meetings),
        "total_action_items": 0,
        "total_decisions": 0,
        "total_participants": set(),
        "total_deadlines": 0,
        "linked_meetings": [],
        "errors": []
    }

    if not meetings:
        report["status"] = "EMPTY"
        report["message"] = "No meeting records found in database."
        return report

    for m in meetings:
        m_info = {
            "id": m["id"],
            "filename": m["filename"],
            "has_transcript": bool(m["transcript"]),
            "has_summary": bool(m["summary"]),
            "decisions_count": len(m["key_decisions"]),
            "action_items_count": len(m["action_items"]),
            "participants_count": len(m["participants"]),
            "deadlines_count": sum(1 for a in m["action_items"] if a.get("deadline"))
        }
        report["total_decisions"] += m_info["decisions_count"]
        report["total_action_items"] += m_info["action_items_count"]
        report["total_deadlines"] += m_info["deadlines_count"]
        for p in m["participants"]:
            report["total_participants"].add(p)
        
        # Verify foreign keys & linking
        if not m["filename"]:
            report["errors"].append(f"Meeting {m['id']} missing filename metadata")
        if not m["transcript"]:
            report["errors"].append(f"Meeting {m['id']} missing transcript")
            
        report["linked_meetings"].append(m_info)

    report["total_participants"] = list(report["total_participants"])
    if report["errors"]:
        report["status"] = "WARNING"

    return report
