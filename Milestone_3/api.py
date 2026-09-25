import os
import time
import logging
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Depends, HTTPException, Query, status, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Re-use existing Milestone 3 services and models directly
from database import (
    init_db,
    create_user,
    authenticate_user,
    get_all_meetings,
    get_user_meetings,
    get_meeting_by_id,
    verify_knowledge_repository
)
from vector_store import get_vector_store_stats
from search_service import semantic_search_meetings, TARGET_SLA_SECONDS
from rag_service import generate_grounded_rag_answer
from schemas import (
    UserRegisterRequest,
    UserLoginRequest,
    AuthResponse,
    SearchRequest,
    AskRequest,
    RAGAnswer,
    ErrorResponse
)

# ─────────────────────────────────────────────
#  LOGGING & SETUP
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("synthai_api")

# Auto-initialize database tables
init_db()

app = FastAPI(
    title="SynthAI — Meeting Intelligence & Semantic RAG API",
    description="Enterprise API connecting relational meeting intelligence, persistent vector search, and grounded RAG question answering.",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security Scheme
security = HTTPBearer(auto_error=False)

# In-memory session tokens for demo / stateless authentication
_ACTIVE_TOKENS: Dict[str, Dict[str, Any]] = {}

def create_access_token(user_id: int, username: str) -> str:
    import secrets
    token = f"synthai_token_{secrets.token_hex(16)}"
    _ACTIVE_TOKENS[token] = {
        "user_id": user_id,
        "username": username,
        "created_at": time.time()
    }
    return token

def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[Dict[str, Any]]:
    """Optional dependency to extract authenticated user from Bearer header."""
    if not credentials:
        return None
    token = credentials.credentials
    user_session = _ACTIVE_TOKENS.get(token)
    if not user_session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_session

def require_current_user(user: Optional[Dict[str, Any]] = Depends(get_current_user)) -> Dict[str, Any]:
    """Strict dependency requiring authenticated user."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required to access this endpoint.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


# ─────────────────────────────────────────────
#  REQUEST LOGGING & TIMING MIDDLEWARE
# ─────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()
    method = request.method
    path = request.url.path
    try:
        response = await call_next(request)
        process_time = (time.perf_counter() - start_time) * 1000
        logger.info(f"{method} {path} completed with status {response.status_code} in {process_time:.2f}ms")
        response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
        return response
    except Exception as exc:
        process_time = (time.perf_counter() - start_time) * 1000
        logger.error(f"{method} {path} failed with unhandled exception in {process_time:.2f}ms: {exc}")
        raise exc


# ─────────────────────────────────────────────
#  STANDARDIZED ERROR HANDLING
# ─────────────────────────────────────────────
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTP {exc.status_code} on {request.url.path}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code, "path": request.url.path}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Validation error on {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Request validation failed", "detail": exc.errors(), "status_code": 422}
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Internal server error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error occurred", "detail": str(exc), "status_code": 500}
    )


# ─────────────────────────────────────────────
#  SYSTEM & HEALTH ENDPOINTS
# ─────────────────────────────────────────────
@app.get("/", tags=["System"])
def root():
    return {
        "service": "SynthAI Meeting Intelligence & Semantic RAG API",
        "version": "3.0.0",
        "documentation": "/docs",
        "status": "online"
    }

@app.get("/health", tags=["System"])
def health_check():
    """System health check verifying database and vector store connectivity."""
    db_report = verify_knowledge_repository()
    v_stats = get_vector_store_stats()
    return {
        "status": "healthy",
        "database": {
            "status": db_report.get("status"),
            "total_meetings": db_report.get("total_meetings", 0)
        },
        "vector_store": {
            "total_vectors": v_stats.get("total_vectors", 0),
            "collection_name": v_stats.get("collection_name")
        }
    }

@app.get("/stats", tags=["System"])
def system_stats():
    """Returns vector store distribution and knowledge repository statistics."""
    v_stats = get_vector_store_stats()
    db_report = verify_knowledge_repository()
    return {
        "vector_store": v_stats,
        "knowledge_repository": {
            "total_meetings": db_report.get("total_meetings", 0),
            "total_decisions": db_report.get("total_decisions", 0),
            "total_action_items": db_report.get("total_action_items", 0),
            "total_participants": len(db_report.get("total_participants", []))
        }
    }


# ─────────────────────────────────────────────
#  AUTHENTICATION ENDPOINTS
# ─────────────────────────────────────────────
@app.post("/auth/register", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED, tags=["Authentication"])
def register(payload: UserRegisterRequest):
    """Register a new user account."""
    if len(payload.username.strip()) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters.")
    if len(payload.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")

    created = create_user(payload.username.strip(), payload.password)
    if not created:
        raise HTTPException(status_code=400, detail="Username already exists.")

    return {"message": "User registered successfully", "username": payload.username}

@app.post("/auth/login", response_model=AuthResponse, tags=["Authentication"])
def login(payload: UserLoginRequest):
    """Authenticate with username and password, returning an access token."""
    user = authenticate_user(payload.username.strip(), payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    token = create_access_token(user["id"], user["username"])
    return AuthResponse(
        access_token=token,
        token_type="bearer",
        user_id=user["id"],
        username=user["username"]
    )


# ─────────────────────────────────────────────
#  MEETINGS ENDPOINTS
# ─────────────────────────────────────────────
@app.get("/meetings", tags=["Meetings"])
def list_meetings(
    user_id: Optional[int] = Query(None, description="Optional user ID filter"),
    start_date: Optional[str] = Query(None, description="Start date ISO filter (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date ISO filter (YYYY-MM-DD)"),
    limit: int = Query(50, ge=1, le=200, description="Max number of meetings to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """
    Retrieve historical meetings with metadata, summaries, and creation timestamps.
    Supports user filtering, date filtering, and pagination.
    """
    # If user_id is not specified, return all meetings
    meetings = get_all_meetings(start_date=start_date, end_date=end_date)
    if user_id:
        meetings = [m for m in meetings if m.get("user_id") == user_id]

    total_count = len(meetings)
    paginated = meetings[offset : offset + limit]

    # Return clean summary item list
    formatted = [
        {
            "id": m["id"],
            "filename": m["filename"],
            "summary": m["summary"],
            "created_at": str(m["created_at"]),
            "user_id": m.get("user_id"),
            "action_items_count": len(m.get("action_items", [])),
            "decisions_count": len(m.get("key_decisions", [])),
            "participants": m.get("participants", [])
        }
        for m in paginated
    ]

    return {
        "total": total_count,
        "limit": limit,
        "offset": offset,
        "meetings": formatted
    }

@app.get("/meetings/{id}", tags=["Meetings"])
def get_meeting(id: int, current_user: Optional[Dict[str, Any]] = Depends(get_current_user)):
    """
    Retrieve complete relational graph for a single meeting by ID:
    transcripts, executive summary, action items with assignees & deadlines,
    key decisions, and participants.
    """
    meeting = get_meeting_by_id(id)
    if not meeting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Meeting with ID {id} not found."
        )
    return meeting


# ─────────────────────────────────────────────
#  SEARCH ENDPOINTS (SEMANTIC SEARCH)
# ─────────────────────────────────────────────
@app.post("/search", tags=["Search"])
def search_meetings_post(payload: SearchRequest):
    """
    Perform natural language semantic search across all historical meetings.
    Supports metadata filtering (doc_type, meeting_id) and date filtering.
    Enforces sub-3-second SLA tracking.
    """
    return semantic_search_meetings(
        query=payload.query,
        top_k=payload.top_k,
        doc_type=payload.doc_type,
        meeting_id=payload.meeting_id,
        start_date=payload.start_date,
        end_date=payload.end_date
    )

@app.get("/search", tags=["Search"])
def search_meetings_get(
    query: str = Query(..., description="Natural language search query"),
    top_k: int = Query(8, ge=1, le=50, description="Max number of items to retrieve"),
    doc_type: Optional[str] = Query(None, description="Filter by entity type (summary, decision, action_item, transcript)"),
    meeting_id: Optional[int] = Query(None, description="Filter by specific meeting ID"),
    start_date: Optional[str] = Query(None, description="Start date filter (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date filter (YYYY-MM-DD)")
):
    """GET variant of semantic search for easy browser testing and integration."""
    return semantic_search_meetings(
        query=query,
        top_k=top_k,
        doc_type=doc_type,
        meeting_id=meeting_id,
        start_date=start_date,
        end_date=end_date
    )


# ─────────────────────────────────────────────
#  RAG QUESTION ANSWERING ENDPOINT
# ─────────────────────────────────────────────
@app.post("/ask", response_model=RAGAnswer, tags=["Question Answering"])
def ask_question(payload: AskRequest):
    """
    Ask a question across the meeting knowledge base.
    Answers are strictly grounded in retrieved meeting facts, citing source meetings,
    decisions, and deadlines.
    """
    res = generate_grounded_rag_answer(
        question=payload.question,
        meeting_id=payload.meeting_id,
        doc_type=payload.doc_type,
        start_date=payload.start_date,
        end_date=payload.end_date,
        top_k=payload.top_k
    )
    return RAGAnswer(
        question=res["question"],
        answer=res["answer"],
        sources=res.get("sources", []),
        retrieval_time_seconds=res.get("retrieval_time_seconds", 0.0),
        total_time_seconds=res.get("total_time_seconds"),
        is_grounded=res.get("is_grounded", False)
    )
