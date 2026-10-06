import os
import logging
from typing import List, Dict, Any, Optional
import chromadb
from config import CHROMA_DB_DIR, COLLECTION_NAME
from database import get_meeting_by_id

logger = logging.getLogger(__name__)

_CHROMA_CLIENT = None
_CHROMA_COLLECTION = None

def get_chroma_client():
    """Initializes or returns the persistent ChromaDB client."""
    global _CHROMA_CLIENT
    if _CHROMA_CLIENT is None:
        os.makedirs(CHROMA_DB_DIR, exist_ok=True)
        logger.info(f"Initializing persistent ChromaDB client at: {CHROMA_DB_DIR}")
        _CHROMA_CLIENT = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    return _CHROMA_CLIENT

def get_vector_collection():
    """Gets or creates the persistent meeting knowledge collection."""
    global _CHROMA_COLLECTION
    if _CHROMA_COLLECTION is None:
        client = get_chroma_client()
        _CHROMA_COLLECTION = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    return _CHROMA_COLLECTION

def sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Ensures all metadata values are valid ChromaDB primitives (str, int, float, bool)."""
    clean = {}
    for k, v in meta.items():
        if v is None:
            clean[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean

def insert_meeting_vectors(meeting_id: int, vector_items: List[Dict[str, Any]]) -> int:
    """Insert embeddings with rich metadata into ChromaDB using upsert."""
    if not vector_items:
        return 0

    try:
        collection = get_vector_collection()
        ids = [item["unit_id"] for item in vector_items]
        embeddings = [item["embedding"] for item in vector_items]
        documents = [item["text"] for item in vector_items]
        metadatas = [sanitize_metadata(item.get("metadata", {})) for item in vector_items]

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        logger.info(f"Inserted/Upserted {len(ids)} vectors for meeting_id={meeting_id}")
        return len(ids)
    except Exception as e:
        logger.error(f"Failed to insert vectors for meeting_id={meeting_id}: {e}")
        return 0

def update_meeting_vectors(meeting_id: int, vector_items: List[Dict[str, Any]]) -> int:
    delete_meeting_vectors(meeting_id)
    return insert_meeting_vectors(meeting_id, vector_items)

def delete_meeting_vectors(meeting_id: int) -> int:
    try:
        collection = get_vector_collection()
        existing = collection.get(where={"meeting_id": meeting_id})
        ids_to_delete = existing.get("ids", []) if existing else []
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            logger.info(f"Deleted {len(ids_to_delete)} vectors for meeting_id={meeting_id}")
            return len(ids_to_delete)
        return 0
    except Exception as e:
        logger.warning(f"Error during delete_meeting_vectors for meeting {meeting_id}: {e}")
        return 0

def similarity_search(
    query_embedding: List[float],
    top_k: int = 8,
    doc_type: Optional[str] = None,
    meeting_id: Optional[int] = None,
    user_id: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Query ChromaDB with dynamic metadata filters."""
    try:
        collection = get_vector_collection()
        if collection.count() == 0:
            return []

        conditions = []
        if doc_type:
            conditions.append({"doc_type": doc_type})
        if meeting_id:
            conditions.append({"meeting_id": meeting_id})
        if user_id is not None:
            conditions.append({"$or": [{"user_id": user_id}, {"user_id": -1}]})

        if len(conditions) == 1:
            where_filter = conditions[0]
        elif len(conditions) > 1:
            where_filter = {"$and": conditions}
        else:
            where_filter = None

        count = collection.count()
        n_results = min(top_k * 3, count)
        if n_results <= 0:
            return []

        raw = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        results = []
        ids = raw.get("ids", [[]])[0]
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        dists = raw.get("distances", [[]])[0]

        for u_id, doc, meta, dist in zip(ids, docs, metas, dists):
            created_at = meta.get("created_at", "")
            if start_date and created_at and created_at[:10] < start_date[:10]:
                continue
            if end_date and created_at and created_at[:10] > end_date[:10]:
                continue

            similarity_score = max(0.0, min(1.0, 1.0 - float(dist)))
            results.append({
                "unit_id": u_id,
                "meeting_id": int(meta.get("meeting_id", 0)),
                "filename": meta.get("meeting_filename", ""),
                "doc_type": meta.get("doc_type", "unknown"),
                "text": doc,
                "score": round(similarity_score, 4),
                "metadata": meta
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    except Exception as e:
        logger.error(f"Error during similarity_search: {e}")
        return []

def metadata_filtering(
    query_embedding: List[float],
    top_k: int = 8,
    doc_type: Optional[str] = None,
    meeting_id: Optional[int] = None,
    user_id: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Milestone 3 & 4 compatibility alias for similarity_search with metadata filtering."""
    return similarity_search(
        query_embedding=query_embedding,
        top_k=top_k,
        doc_type=doc_type,
        meeting_id=meeting_id,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date
    )

def get_vector_store_stats() -> Dict[str, Any]:
    try:
        collection = get_vector_collection()
        total = collection.count()
        doc_type_counts = {"summary": 0, "decision": 0, "action_item": 0, "transcript": 0, "other": 0}
        unique_meetings = set()
        if total > 0:
            all_meta = collection.get(include=["metadatas"])
            for m in all_meta.get("metadatas", []):
                dt = m.get("doc_type", "other")
                if dt in doc_type_counts:
                    doc_type_counts[dt] += 1
                else:
                    doc_type_counts["other"] += 1
                m_id = m.get("meeting_id")
                if m_id is not None:
                    unique_meetings.add(m_id)

        return {
            "status": "connected",
            "total_vectors": total,
            "unique_meetings_indexed": len(unique_meetings),
            "collection_name": COLLECTION_NAME,
            "storage_path": CHROMA_DB_DIR,
            "breakdown": doc_type_counts,
            "breakdown_by_doc_type": doc_type_counts
        }
    except Exception as e:
        return {
            "status": "error",
            "total_vectors": 0,
            "unique_meetings_indexed": 0,
            "collection_name": COLLECTION_NAME,
            "storage_path": CHROMA_DB_DIR,
            "breakdown": {},
            "breakdown_by_doc_type": {},
            "error": str(e)
        }

def trace_vector_to_meeting(unit_id: str) -> Optional[Dict[str, Any]]:
    try:
        collection = get_vector_collection()
        res = collection.get(ids=[unit_id], include=["metadatas", "documents"])
        if not res or not res.get("ids"):
            return {
                "traced": False,
                "verified_link": False,
                "unit_id": unit_id,
                "vector_id": unit_id,
                "meeting_id": None,
                "meeting_filename": None,
                "meeting": None
            }
        meta = res["metadatas"][0]
        m_id = int(meta.get("meeting_id", 0))
        meeting = get_meeting_by_id(m_id)
        return {
            "traced": bool(meeting is not None),
            "verified_link": bool(meeting is not None),
            "unit_id": unit_id,
            "vector_id": unit_id,
            "doc_type": meta.get("doc_type"),
            "meeting_id": m_id,
            "meeting_filename": meeting.get("filename") if meeting else meta.get("meeting_filename"),
            "meeting": meeting
        }
    except Exception as e:
        logger.error(f"Failed to trace vector {unit_id}: {e}")
        return {
            "traced": False,
            "verified_link": False,
            "unit_id": unit_id,
            "vector_id": unit_id,
            "error": str(e)
        }

def verify_vector_database_integration() -> Dict[str, Any]:
    """
    Milestone 3 Task 3 Verification:
    Tests Insert, Similarity Search, Metadata Filtering, Update, and Delete.
    """
    test_m_id = 999999
    sample_items = [
        {
            "unit_id": f"m_{test_m_id}_summary",
            "meeting_id": test_m_id,
            "text": "Temporary test meeting executive summary for verification.",
            "embedding": [0.05] * 384,
            "metadata": {"meeting_id": test_m_id, "doc_type": "summary", "meeting_filename": "test.mp4"}
        },
        {
            "unit_id": f"m_{test_m_id}_action_0",
            "meeting_id": test_m_id,
            "text": "Temporary action item: verify vector database integration.",
            "embedding": [0.08] * 384,
            "metadata": {"meeting_id": test_m_id, "doc_type": "action_item", "deadline": "Immediate", "meeting_filename": "test.mp4"}
        }
    ]

    checks = {}
    try:
        inserted = insert_meeting_vectors(test_m_id, sample_items)
        checks["insert"] = (inserted == 2)
    except Exception as e:
        checks["insert"] = False

    try:
        search_res = similarity_search([0.05] * 384, top_k=2)
        checks["similarity_search"] = (len(search_res) > 0)
    except Exception as e:
        checks["similarity_search"] = False

    try:
        filtered_res = metadata_filtering([0.08] * 384, top_k=5, doc_type="action_item", meeting_id=test_m_id)
        checks["metadata_filtering"] = (len(filtered_res) == 1 and filtered_res[0]["doc_type"] == "action_item")
    except Exception as e:
        checks["metadata_filtering"] = False

    try:
        sample_items[1]["text"] = "Updated action item text."
        updated = update_meeting_vectors(test_m_id, sample_items)
        checks["update"] = (updated == 2)
    except Exception as e:
        checks["update"] = False

    try:
        deleted = delete_meeting_vectors(test_m_id)
        checks["delete"] = (deleted == 2)
    except Exception as e:
        checks["delete"] = False

    overall = all(checks.get(k) is True for k in ["insert", "similarity_search", "metadata_filtering", "update", "delete"])
    return {
        "status": "PASS" if overall else "FAIL",
        "checks": checks
    }
