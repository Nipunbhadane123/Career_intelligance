import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Table, Float
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()

# Association table for meeting and participants
meeting_participant_association = Table(
    'meeting_participant',
    Base.metadata,
    Column('meeting_id', Integer, ForeignKey('meetings.id', ondelete="CASCADE"), primary_key=True),
    Column('participant_id', Integer, ForeignKey('participants.id', ondelete="CASCADE"), primary_key=True)
)

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), default="member")
    is_active = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # User owns meetings
    meetings = relationship("Meeting", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user", cascade="all, delete-orphan")

class Meeting(Base):
    __tablename__ = "meetings"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), index=True, nullable=False)
    title = Column(String(255), nullable=True)
    transcript = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    duration_seconds = Column(Float, default=0.0)
    platform = Column(String(50), default="upload")  # 'upload', 'zoom', 'google_meet'
    external_id = Column(String(255), nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)

    # Relationships
    user = relationship("User", back_populates="meetings")
    action_items = relationship("ActionItem", back_populates="meeting", cascade="all, delete-orphan")
    key_points = relationship("KeyPoint", back_populates="meeting", cascade="all, delete-orphan")
    key_decisions = relationship("KeyDecision", back_populates="meeting", cascade="all, delete-orphan")
    participants = relationship("Participant", secondary=meeting_participant_association, back_populates="meetings")
    sync_records = relationship("IntegrationSync", back_populates="meeting", cascade="all, delete-orphan")

class Participant(Base):
    __tablename__ = "participants"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(150), unique=True, index=True, nullable=False)

    meetings = relationship("Meeting", secondary=meeting_participant_association, back_populates="participants")
    action_items = relationship("ActionItem", back_populates="participant")

class ActionItem(Base):
    __tablename__ = "action_items"

    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(Integer, ForeignKey("meetings.id", ondelete="CASCADE"), index=True, nullable=False)
    participant_id = Column(Integer, ForeignKey("participants.id", ondelete="SET NULL"), nullable=True)
    description = Column(String(500), nullable=False)
    deadline = Column(String(100), nullable=True)
    priority = Column(String(50), default="Medium")  # High, Medium, Low
    status = Column(String(50), default="Pending")    # Pending, In Progress, Completed

    meeting = relationship("Meeting", back_populates="action_items")
    participant = relationship("Participant", back_populates="action_items")

class KeyPoint(Base):
    __tablename__ = "key_points"

    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(Integer, ForeignKey("meetings.id", ondelete="CASCADE"), index=True, nullable=False)
    point = Column(String(500), nullable=False)

    meeting = relationship("Meeting", back_populates="key_points")

class KeyDecision(Base):
    __tablename__ = "key_decisions"

    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(Integer, ForeignKey("meetings.id", ondelete="CASCADE"), index=True, nullable=False)
    decision = Column(String(500), nullable=False)

    meeting = relationship("Meeting", back_populates="key_decisions")

class IntegrationSync(Base):
    __tablename__ = "integration_syncs"

    id = Column(Integer, primary_key=True, index=True)
    platform = Column(String(50), nullable=False, index=True)  # 'zoom' or 'google_meet'
    external_id = Column(String(255), nullable=False, index=True)  # recording UUID or Drive file ID
    file_name = Column(String(255), nullable=False)
    file_hash = Column(String(64), nullable=True, index=True)      # SHA-256 for duplicate detection
    status = Column(String(50), default="pending")                 # 'pending', 'completed', 'duplicate', 'failed'
    meeting_id = Column(Integer, ForeignKey("meetings.id", ondelete="SET NULL"), nullable=True)
    error_message = Column(Text, nullable=True)
    synced_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)

    meeting = relationship("Meeting", back_populates="sync_records")

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    action = Column(String(100), nullable=False)
    resource = Column(String(255), nullable=False)
    status = Column(String(50), default="SUCCESS")  # 'SUCCESS', 'DENIED', 'ERROR'
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="audit_logs")
