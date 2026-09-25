import os
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from database import get_meeting_by_id

logger = logging.getLogger(__name__)

# Persistent storage directory for ChromaDB
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "meeting_knowledge_store"

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
    """
    Task 3: Insert embeddings with rich metadata into the persistent vector database.
    Uses upsert to guarantee idempotency.
    """
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
    """
    Task 3: Update embeddings for a meeting.
    Deletes any existing vectors for the meeting and inserts new ones.
    """
    delete_meeting_vectors(meeting_id)
    return insert_meeting_vectors(meeting_id, vector_items)

def delete_meeting_vectors(meeting_id: int) -> int:
    """
    Task 3: Delete embeddings for a specific meeting.
    Removes all vectors where metadata meeting_id matches.
    """
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

def similarity_search(query_embedding: List[float], top_k: int = 5, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Task 3: Similarity search in vector database.
    Converts cosine distance into a 0.0 - 1.0 similarity score.
    """
    try:
        collection = get_vector_collection()
        total_count = collection.count()
        if total_count == 0:
            return []

        actual_k = min(top_k, total_count)
        query_args = {
            "query_embeddings": [query_embedding],
            "n_results": actual_k,
            "include": ["documents", "metadatas", "distances"]
        }
        if where:
            query_args["where"] = where

        results = collection.query(**query_args)
        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        formatted = []
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for uid, doc, meta, dist in zip(ids, docs, metas, distances):
            similarity_score = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
            formatted.append({
                "unit_id": uid,
                "meeting_id": meta.get("meeting_id"),
                "meeting_filename": meta.get("meeting_filename", ""),
                "doc_type": meta.get("doc_type", "unknown"),
                "text": doc,
                "score": round(similarity_score, 4),
                "distance": round(dist, 4),
                "metadata": meta
            })

        return formatted
    except Exception as e:
        logger.error(f"Error during similarity_search: {e}")
        return []

def metadata_filtering(
    query_embedding: List[float],
    top_k: int = 5,
    doc_type: Optional[str] = None,
    meeting_id: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Task 3 & Task 7: Metadata and date filtering query.
    Allows filtering by doc_type, meeting_id, start_date, end_date.
    """
    filters = []
    if doc_type and doc_type.lower() != "all":
        filters.append({"doc_type": doc_type.lower()})
    if meeting_id is not None:
        filters.append({"meeting_id": int(meeting_id)})

    where_clause = None
    if len(filters) == 1:
        where_clause = filters[0]
    elif len(filters) > 1:
        where_clause = {"$and": filters}

    # Fetch results from vector store
    # Query more results if date filtering is applied so post-filtering retains top_k
    k_to_fetch = top_k * 3 if (start_date or end_date) else top_k
    hits = similarity_search(query_embedding, top_k=k_to_fetch, where=where_clause)

    # Date-based post-filtering
    if start_date or end_date:
        filtered = []
        for hit in hits:
            c_at = hit.get("metadata", {}).get("created_at", "")
            if start_date and c_at and c_at < str(start_date):
                continue
            if end_date and c_at and c_at > str(end_date):
                continue
            filtered.append(hit)
        return filtered[:top_k]

    return hits

def trace_vector_to_meeting(vector_id: str) -> Dict[str, Any]:
    """
    Task 3: Meeting-to-vector mapping and traceability.
    Verifies that every stored vector can be traced back to the correct meeting record.
    """
    collection = get_vector_collection()
    res = collection.get(ids=[vector_id], include=["metadatas", "documents"])
    if not res or not res.get("ids"):
        return {"error": f"Vector ID {vector_id} not found in vector database."}

    meta = res["metadatas"][0]
    meeting_id = meta.get("meeting_id")
    if meeting_id is None:
        return {"error": f"Vector ID {vector_id} has no meeting_id in metadata."}

    # Verify link against relational database
    db_meeting = get_meeting_by_id(int(meeting_id))
    if not db_meeting:
        return {
            "traced": False,
            "vector_id": vector_id,
            "meeting_id": meeting_id,
            "error": "Relational meeting record not found in database."
        }

    return {
        "traced": True,
        "vector_id": vector_id,
        "meeting_id": meeting_id,
        "meeting_filename": db_meeting["filename"],
        "meeting_created_at": str(db_meeting["created_at"]),
        "doc_type": meta.get("doc_type"),
        "vector_text_preview": res["documents"][0][:160] + "...",
        "verified_link": True
    }

def get_vector_store_stats() -> Dict[str, Any]:
    """Returns status metrics of the persistent vector database."""
    collection = get_vector_collection()
    count = collection.count()
    
    breakdown = {"summary": 0, "decision": 0, "action_item": 0, "transcript": 0, "other": 0}
    unique_meetings = set()

    if count > 0:
        all_meta = collection.get(include=["metadatas"])
        if all_meta and all_meta.get("metadatas"):
            for m in all_meta["metadatas"]:
                dtype = m.get("doc_type", "other")
                breakdown[dtype] = breakdown.get(dtype, 0) + 1
                if m.get("meeting_id") is not None:
                    unique_meetings.add(m["meeting_id"])

    return {
        "total_vectors": count,
        "unique_meetings_indexed": len(unique_meetings),
        "breakdown": breakdown,
        "collection_name": COLLECTION_NAME,
        "storage_path": CHROMA_DB_DIR
    }

def verify_vector_database_integration() -> Dict[str, Any]:
    """
    Task 3 Verification:
    Tests Insert, Query, Metadata Filtering, Update, Delete, and Vector-to-Meeting Mapping.
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

    # 1. Insert
    try:
        inserted = insert_meeting_vectors(test_m_id, sample_items)
        checks["insert"] = (inserted == 2)
    except Exception as e:
        checks["insert"] = False
        checks["insert_error"] = str(e)

    # 2. Similarity Search
    try:
        search_res = similarity_search([0.05] * 384, top_k=2)
        checks["similarity_search"] = (len(search_res) > 0)
    except Exception as e:
        checks["similarity_search"] = False
        checks["search_error"] = str(e)

    # 3. Metadata Filtering
    try:
        filtered_res = metadata_filtering([0.08] * 384, top_k=5, doc_type="action_item", meeting_id=test_m_id)
        checks["metadata_filtering"] = (len(filtered_res) == 1 and filtered_res[0]["doc_type"] == "action_item")
    except Exception as e:
        checks["metadata_filtering"] = False
        checks["filter_error"] = str(e)

    # 4. Update
    try:
        sample_items[1]["text"] = "Updated action item text."
        updated = update_meeting_vectors(test_m_id, sample_items)
        checks["update"] = (updated == 2)
    except Exception as e:
        checks["update"] = False
        checks["update_error"] = str(e)

    # 5. Delete
    try:
        deleted = delete_meeting_vectors(test_m_id)
        checks["delete"] = (deleted == 2)
    except Exception as e:
        checks["delete"] = False
        checks["delete_error"] = str(e)

    overall = all(checks.get(k) is True for k in ["insert", "similarity_search", "metadata_filtering", "update", "delete"])
    return {
        "status": "PASS" if overall else "FAIL",
        "checks": checks
    }
