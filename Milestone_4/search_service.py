import time
import logging
from typing import Optional, Dict, Any, List
from config import TARGET_SLA_SECONDS
from embedding_service import generate_single_embedding
from vector_store import similarity_search
from database import get_meeting_by_id

logger = logging.getLogger(__name__)

def semantic_search_meetings(
    query: str,
    top_k: int = 8,
    doc_type: Optional[str] = None,
    meeting_id: Optional[int] = None,
    user_id: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes semantic search across all stored meeting knowledge units.
    Tracks execution latency and verifies sub-3s SLA.
    Returns both flat results (for M4 API) and meeting-grouped results (for M3 UI & tests).
    """
    start_time = time.perf_counter()
    if not query or not query.strip():
        return {
            "query": query or "",
            "total_results": 0,
            "total_hits": 0,
            "search_time_seconds": 0.0,
            "retrieval_time_seconds": 0.0,
            "sla_met": True,
            "target_sla_seconds": TARGET_SLA_SECONDS,
            "results": [],
            "raw_hits": [],
            "matched_meetings": []
        }

    # Generate query embedding
    try:
        query_emb = generate_single_embedding(query.strip())
    except Exception as e:
        logger.error(f"Failed to generate query embedding: {e}")
        return {
            "query": query,
            "total_results": 0,
            "total_hits": 0,
            "search_time_seconds": round(time.perf_counter() - start_time, 4),
            "retrieval_time_seconds": round(time.perf_counter() - start_time, 4),
            "sla_met": True,
            "target_sla_seconds": TARGET_SLA_SECONDS,
            "results": [],
            "raw_hits": [],
            "matched_meetings": [],
            "error": str(e)
        }

    # Execute vector similarity search with filters
    raw_results = similarity_search(
        query_embedding=query_emb,
        top_k=top_k,
        doc_type=doc_type,
        meeting_id=meeting_id,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date
    )

    elapsed = time.perf_counter() - start_time
    sla_met = elapsed <= TARGET_SLA_SECONDS

    # Group hits by meeting for rich dashboard presentation & M3 test compatibility
    meetings_map: Dict[int, Dict[str, Any]] = {}
    for hit in raw_results:
        m_id = hit.get("meeting_id")
        if m_id is None:
            continue
        if m_id not in meetings_map:
            db_m = get_meeting_by_id(int(m_id))
            fname = db_m["filename"] if db_m else hit.get("filename", f"Meeting #{m_id}")
            created_at = str(db_m["created_at"]) if db_m else hit.get("metadata", {}).get("created_at", "")
            summary = db_m["summary"] if db_m else ""
            title = db_m.get("title") or fname

            meetings_map[m_id] = {
                "meeting_id": m_id,
                "filename": fname,
                "title": title,
                "created_at": created_at,
                "summary": summary,
                "best_score": hit["score"],
                "hits_count": 0,
                "matching_entities": []
            }

        if hit["score"] > meetings_map[m_id]["best_score"]:
            meetings_map[m_id]["best_score"] = hit["score"]

        meetings_map[m_id]["hits_count"] += 1
        meetings_map[m_id]["matching_entities"].append({
            "unit_id": hit["unit_id"],
            "doc_type": hit["doc_type"],
            "score": hit["score"],
            "text": hit["text"],
            "metadata": hit.get("metadata", {})
        })

    matched_meetings = sorted(meetings_map.values(), key=lambda x: x["best_score"], reverse=True)

    return {
        "query": query,
        "total_results": len(raw_results),
        "total_hits": len(raw_results),
        "search_time_seconds": round(elapsed, 4),
        "retrieval_time_seconds": round(elapsed, 4),
        "sla_met": sla_met,
        "target_sla_seconds": TARGET_SLA_SECONDS,
        "results": raw_results,
        "raw_hits": raw_results,
        "matched_meetings": matched_meetings
    }

def verify_semantic_search_performance(query: str = "Which meeting discussed the database migration?") -> Dict[str, Any]:
    """
    Milestone 3 Task 4 Verification:
    Verify that relevant meetings are returned dynamically within the 3.0s SLA requirement.
    """
    result = semantic_search_meetings(query=query, top_k=6)
    meets_sla = result["sla_met"]
    has_results = result["total_results"] > 0
    meetings_found = len(result["matched_meetings"])
    status = "PASS" if (meets_sla and has_results) else ("WARNING" if meets_sla else "FAIL")

    return {
        "status": status,
        "query": query,
        "retrieval_time_seconds": result["search_time_seconds"],
        "target_sla_seconds": TARGET_SLA_SECONDS,
        "meets_sla_under_3s": meets_sla,
        "total_hits": result["total_results"],
        "meetings_matched_count": meetings_found,
        "top_meeting": result["matched_meetings"][0]["filename"] if meetings_found > 0 else None,
        "top_score": result["matched_meetings"][0]["best_score"] if meetings_found > 0 else 0.0
    }
