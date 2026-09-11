import os
import bcrypt
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Use SQLite for persistence, stored locally in the current directory
DB_FILE = "meeting_intelligence.db"
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_FILE}"

# check_same_thread=False is needed for SQLite in Streamlit/multi-threaded environments
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def init_db():
    import models  # import models here to ensure they are registered with Base
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
        # Check if user already exists
        if db.query(User).filter(User.username == username).first():
            return False
            
        # Hash password
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
            # Return a plain dict to avoid detached ORM object errors
            return {"id": user.id, "username": user.username}
        return None
    finally:
        db.close()

def get_user_meetings(user_id: int = None):
    from models import Meeting
    db = SessionLocal()
    try:
        query = db.query(Meeting)
        meetings = query.order_by(Meeting.created_at.desc()).all()
        return [(m.id, m.filename, m.transcript, m.summary, m.created_at) for m in meetings]
    finally:
        db.close()

def get_meeting_by_id(meeting_id: int):
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
            "action_items": [{"description": a.description, "assigned_participant": a.participant.name if a.participant else (a.participant_id or "Unassigned"), "deadline": a.deadline, "priority": a.priority, "status": a.status} for a in m.action_items],
            "key_points": [kp.point for kp in m.key_points],
            "key_decisions": [kd.decision for kd in m.key_decisions],
            "participants": [p.name for p in m.participants]
        }
    finally:
        db.close()

