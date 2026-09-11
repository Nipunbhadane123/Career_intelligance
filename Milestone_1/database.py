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
            return user
        return None
    finally:
        db.close()

def save_meeting(user_id: int, filename: str, transcript: str, summary: str):
    from models import Meeting
    db = SessionLocal()
    try:
        meeting = Meeting(
            filename=filename,
            transcript=transcript,
            summary=summary,
            user_id=user_id,
        )
        db.add(meeting)
        db.commit()
        db.refresh(meeting)
        return meeting.id
    finally:
        db.close()

def get_user_meetings(user_id: int):
    from models import Meeting
    db = SessionLocal()
    try:
        meetings = db.query(Meeting).filter(Meeting.user_id == user_id).order_by(Meeting.created_at.desc()).all()
        return [(m.id, m.filename, m.transcript, m.summary, m.created_at) for m in meetings]
    finally:
        db.close()
