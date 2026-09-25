import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from database import export_meeting_knowledge_units

logger = logging.getLogger(__name__)

# Global cached embedding model instance
_EMBEDDING_MODEL = None
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

def get_embedding_model() -> SentenceTransformer:
    """Load or return the cached SentenceTransformer model."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        try:
            # First attempt loading from local cache to avoid network roundtrips & timeout delays
            _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
        except Exception:
            _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _EMBEDDING_MODEL

def generate_single_embedding(text: str) -> List[float]:
    """Generate a single 384-dimensional dense embedding vector."""
    if not text or not text.strip():
        text = "Empty document"
    model = get_embedding_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()

def generate_batch_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate dense embeddings for a batch of texts."""
    if not texts:
        return []
    cleaned = [t if (t and t.strip()) else "Empty document" for t in texts]
    model = get_embedding_model()
    embeddings = model.encode(cleaned, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.tolist()

def generate_meeting_embeddings(meeting_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Task 2 Core Engine:
    Dynamically generates embeddings for meeting information across all 4 entity types:
    1. Transcript sections
    2. Summaries
    3. Decisions
    4. Action items (with deadlines and participants)

    Returns a list of structured items ready for vector database insertion.
    Each item contains:
    - unit_id: deterministic ID
    - meeting_id: integer meeting foreign key
    - doc_type: 'transcript' | 'summary' | 'decision' | 'action_item'
    - title: readable heading
    - text: full document text
    - metadata: dictionary containing meeting_id, filename, doc_type, deadlines, etc.
    - embedding: 384-dimensional float vector
    """
    units = export_meeting_knowledge_units(meeting_dict)
    if not units:
        return []

    texts = [u["text"] for u in units]
    embeddings = generate_batch_embeddings(texts)

    for unit, emb in zip(units, embeddings):
        unit["embedding"] = emb

    return units

def verify_embedding_generation(meeting_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Task 2 Verification:
    Verify that embeddings are generated dynamically and correctly linked to meeting records.
    """
    results = generate_meeting_embeddings(meeting_dict)
    
    doc_types_found = set()
    total_embeddings = len(results)
    valid_dimension = True
    all_linked = True
    m_id = meeting_dict.get("id")

    for item in results:
        doc_types_found.add(item["doc_type"])
        if len(item["embedding"]) != EMBEDDING_DIM:
            valid_dimension = False
        if item["meeting_id"] != m_id or item["metadata"].get("meeting_id") != m_id:
            all_linked = False

    expected_types = {"summary", "decision", "action_item", "transcript"}
    # Check which expected types were generated based on meeting contents
    has_summary = bool(meeting_dict.get("summary"))
    has_decisions = bool(meeting_dict.get("key_decisions"))
    has_actions = bool(meeting_dict.get("action_items"))
    has_transcript = bool(meeting_dict.get("transcript"))

    status = "PASS" if (total_embeddings > 0 and valid_dimension and all_linked) else "FAIL"

    return {
        "status": status,
        "meeting_id": m_id,
        "meeting_filename": meeting_dict.get("filename"),
        "total_embeddings_generated": total_embeddings,
        "embedding_dimension": EMBEDDING_DIM,
        "valid_dimension": valid_dimension,
        "all_linked_to_meeting": all_linked,
        "doc_types_generated": list(doc_types_found),
        "coverage": {
            "summary_embedded": "summary" in doc_types_found if has_summary else "N/A",
            "decisions_embedded": "decision" in doc_types_found if has_decisions else "N/A",
            "action_items_embedded": "action_item" in doc_types_found if has_actions else "N/A",
            "transcript_sections_embedded": "transcript" in doc_types_found if has_transcript else "N/A"
        }
    }
