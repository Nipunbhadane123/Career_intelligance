from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

# ─────────────────────────────────────────────
# AUTH SCHEMAS
# ─────────────────────────────────────────────
class UserRegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, description="Unique username")
    password: str = Field(..., min_length=6, description="Password (at least 6 characters)")
    role: Optional[str] = Field("member", description="User role ('member' or 'admin')")

class UserLoginRequest(BaseModel):
    username: str = Field(..., description="Registered username")
    password: str = Field(..., description="User password")

class UserResponse(BaseModel):
    id: int
    username: str
    role: str
    created_at: datetime

    class Config:
        from_attributes = True

class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    username: str
    role: str = "member"

# ─────────────────────────────────────────────
# MEETING SCHEMAS
# ─────────────────────────────────────────────
class ActionItemSchema(BaseModel):
    id: Optional[int] = None
    description: str
    assigned_participant: str = "Unassigned"
    deadline: Optional[str] = None
    priority: str = "Medium"
    status: str = "Pending"

class KeyDecisionSchema(BaseModel):
    id: Optional[int] = None
    decision: str

class MeetingIntelligenceSchema(BaseModel):
    summary: str = Field(description="A 1-2 paragraph executive summary of the meeting.")
    key_points: List[str] = Field(description="A list of key discussion points from the meeting.")
    decisions: List[str] = Field(description="A list of key decisions made during the meeting.")
    action_items: List[ActionItemSchema] = Field(description="A list of extracted action items.")
    participants: List[str] = Field(description="A list of all unique participant names identified in the meeting.")

class MeetingSummaryResponse(BaseModel):
    id: int
    filename: str
    title: Optional[str] = None
    summary: Optional[str] = None
    duration_seconds: float = 0.0
    platform: str = "upload"
    created_at: str
    user_id: Optional[int] = None
    action_items_count: int = 0
    decisions_count: int = 0
    participants: List[str] = []

class MeetingDetailResponse(BaseModel):
    id: int
    filename: str
    title: Optional[str] = None
    transcript: Optional[str] = None
    summary: Optional[str] = None
    duration_seconds: float = 0.0
    platform: str = "upload"
    external_id: Optional[str] = None
    created_at: str
    user_id: Optional[int] = None
    action_items: List[ActionItemSchema] = []
    key_points: List[str] = []
    key_decisions: List[str] = []
    participants: List[str] = []

class MeetingListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    meetings: List[MeetingSummaryResponse]

# ─────────────────────────────────────────────
# SEARCH & RAG SCHEMAS
# ─────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language search query")
    top_k: int = Field(8, ge=1, le=50, description="Max results")
    doc_type: Optional[str] = Field(None, description="summary, decision, action_item, transcript")
    meeting_id: Optional[int] = Field(None, description="Specific meeting ID")
    start_date: Optional[str] = Field(None, description="Start date filter YYYY-MM-DD")
    end_date: Optional[str] = Field(None, description="End date filter YYYY-MM-DD")

class SearchResultItem(BaseModel):
    unit_id: str
    meeting_id: int
    filename: str
    doc_type: str
    text: str
    score: float
    metadata: Dict[str, Any] = {}

class SearchResponse(BaseModel):
    query: str
    total_results: int
    search_time_seconds: float
    sla_met: bool
    results: List[SearchResultItem]
    total_hits: Optional[int] = None
    retrieval_time_seconds: Optional[float] = None
    matched_meetings: Optional[List[Dict[str, Any]]] = None
    raw_hits: Optional[List[Dict[str, Any]]] = None

class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, description="Natural language question")
    meeting_id: Optional[int] = Field(None, description="Optional meeting ID filter")
    doc_type: Optional[str] = Field(None, description="Optional entity type filter")
    start_date: Optional[str] = Field(None, description="Optional start date")
    end_date: Optional[str] = Field(None, description="Optional end date")
    top_k: int = Field(5, ge=1, le=20)

class RAGSource(BaseModel):
    meeting_id: int
    filename: str
    doc_type: str
    preview: str
    relevance_score: float
    date: Optional[str] = None

class RAGAnswer(BaseModel):
    question: str
    answer: str
    sources: List[RAGSource] = []
    retrieval_time_seconds: float = 0.0
    total_time_seconds: Optional[float] = None
    is_grounded: bool = True

# ─────────────────────────────────────────────
# INTEGRATION SCHEMAS (ZOOM & GOOGLE MEET)
# ─────────────────────────────────────────────
class ZoomRecordingItem(BaseModel):
    meeting_id: str
    topic: str
    start_time: str
    duration_minutes: int
    file_type: str
    file_size_mb: float
    download_url: Optional[str] = None
    status: str = "available"

class ZoomSyncRequest(BaseModel):
    recording_id: str = Field(..., description="Zoom recording UUID or ID")
    topic: Optional[str] = None
    force_resync: bool = False

class ZoomSyncResponse(BaseModel):
    status: str
    recording_id: str
    meeting_id: Optional[int] = None
    message: str
    is_duplicate: bool = False

class GoogleMeetRecordingItem(BaseModel):
    file_id: str
    file_name: str
    created_time: str
    size_mb: float
    mime_type: str
    status: str = "available"

class GoogleMeetSyncRequest(BaseModel):
    file_id: str = Field(..., description="Google Drive file ID for Meet recording")
    file_name: Optional[str] = None
    force_resync: bool = False

class GoogleMeetSyncResponse(BaseModel):
    status: str
    file_id: str
    meeting_id: Optional[int] = None
    message: str
    is_duplicate: bool = False

# ─────────────────────────────────────────────
# REPORT & ANALYTICS SCHEMAS
# ─────────────────────────────────────────────
class ReportExportResponse(BaseModel):
    meeting_id: int
    format: str
    filename: str
    download_url: str
    file_size_bytes: int
    generated_at: str

class MeetingAnalyticsResponse(BaseModel):
    meeting_id: int
    filename: str
    duration_seconds: float
    word_count: int
    participant_count: int
    action_items_total: int
    action_items_by_priority: Dict[str, int]
    action_items_by_status: Dict[str, int]
    decisions_count: int
    participant_contributions: Dict[str, int]

class UserAnalyticsOverview(BaseModel):
    total_meetings: int
    total_duration_hours: float
    total_action_items: int
    completed_action_items: int
    pending_action_items: int
    completion_rate_percent: float = 0.0
    total_decisions: int
    total_participants: int
    unique_participants_count: int = 0
    platform_breakdown: Dict[str, int]

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[Any] = None
    status_code: int
    path: Optional[str] = None
