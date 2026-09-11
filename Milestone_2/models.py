from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Table
from sqlalchemy.orm import relationship
import datetime
from database import Base

# Association table for meeting and participants
meeting_participant_association = Table(
    'meeting_participant',
    Base.metadata,
    Column('meeting_id', Integer, ForeignKey('meetings.id')),
    Column('participant_id', Integer, ForeignKey('participants.id'))
)

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password_hash = Column(String)

    meetings = relationship("Meeting", back_populates="user", cascade="all, delete-orphan")

class Meeting(Base):
    __tablename__ = "meetings"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    transcript = Column(Text)
    summary = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    user_id = Column(Integer, ForeignKey("users.id"))

    # Relationships
    user = relationship("User", back_populates="meetings")
    action_items = relationship("ActionItem", back_populates="meeting", cascade="all, delete-orphan")
    key_points = relationship("KeyPoint", back_populates="meeting", cascade="all, delete-orphan")
    key_decisions = relationship("KeyDecision", back_populates="meeting", cascade="all, delete-orphan")
    participants = relationship("Participant", secondary=meeting_participant_association, back_populates="meetings")

class Participant(Base):
    __tablename__ = "participants"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)

    meetings = relationship("Meeting", secondary=meeting_participant_association, back_populates="participants")
    action_items = relationship("ActionItem", back_populates="participant")

class ActionItem(Base):
    __tablename__ = "action_items"

    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(Integer, ForeignKey("meetings.id"))
    participant_id = Column(Integer, ForeignKey("participants.id"), nullable=True) # Could be unassigned
    description = Column(String)
    deadline = Column(String, nullable=True)
    priority = Column(String, nullable=True)
    status = Column(String, default="Pending")

    meeting = relationship("Meeting", back_populates="action_items")
    participant = relationship("Participant", back_populates="action_items")

class KeyPoint(Base):
    __tablename__ = "key_points"

    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(Integer, ForeignKey("meetings.id"))
    point = Column(String)

    meeting = relationship("Meeting", back_populates="key_points")

class KeyDecision(Base):
    __tablename__ = "key_decisions"

    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(Integer, ForeignKey("meetings.id"))
    decision = Column(String)

    meeting = relationship("Meeting", back_populates="key_decisions")
