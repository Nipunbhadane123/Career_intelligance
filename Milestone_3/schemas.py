from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ActionItemSchema(BaseModel):
    description: str = Field(description="The actual task or action item description.")
    assigned_participant: Optional[str] = Field(default="Unknown", description="The name of the participant assigned to the action item. Use 'Unknown' if not explicitly stated.")
    deadline: Optional[str] = Field(default=None, description="The deadline or timeframe for the task, if specified.")
    priority: Optional[str] = Field(default="Medium", description="The priority of the task (e.g., High, Medium, Low), if inferable.")
    status: Optional[str] = Field(default="Pending", description="The status of the task. Usually 'Pending'.")

class MeetingIntelligenceSchema(BaseModel):
    summary: str = Field(description="A 1-2 paragraph executive summary of the meeting.")
    key_points: List[str] = Field(description="A list of key discussion points from the meeting.")
    decisions: List[str] = Field(description="A list of key decisions made during the meeting.")
    action_items: List[ActionItemSchema] = Field(description="A list of extracted action items.")
    participants: List[str] = Field(description="A list of all unique participant names identified in the meeting.")

class SearchResultItem(BaseModel):
    unit_id: str
    meeting_id: int
    meeting_filename: str
    doc_type: str
    title: str
    snippet: str
    score: float
    metadata: Dict[str, Any]

class RAGAnswer(BaseModel):
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    retrieval_time_seconds: float
    total_time_seconds: Optional[float] = None
    is_grounded: bool = True

class UserRegisterRequest(BaseModel):
    username: str
    password: str

class UserLoginRequest(BaseModel):
    username: str
    password: str

class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    username: str

class SearchRequest(BaseModel):
    query: str
    top_k: int = 8
    doc_type: Optional[str] = None
    meeting_id: Optional[int] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    meeting_id: Optional[int] = None
    doc_type: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[Any] = None
    status_code: int

