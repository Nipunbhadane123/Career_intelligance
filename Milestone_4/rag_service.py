import os
import time
import logging
from typing import Dict, Any, Optional, List
from google import genai
from config import GEMINI_API_KEY
from search_service import semantic_search_meetings

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """You are SynthAI's Expert Meeting Intelligence Assistant.
Answer the user's question based STRICTLY and ONLY on the provided meeting context below.

Guiding Principles:
1. Ground your answer directly in the provided meeting context.
2. Explicitly cite the source meeting filename, the type of information (e.g., Key Decision, Action Item, Executive Summary, or Transcript Segment), and any dates/deadlines mentioned.
3. If the retrieved context does not contain the answer, reply: "Based on the historical meetings in the knowledge base, I do not have enough information to answer this question."
4. Do NOT speculate, hallucinate, or use external unsupported facts.

--- RETRIEVED MEETING CONTEXT ---
{context}
---------------------------------
"""

def assemble_context(raw_hits: List[Dict[str, Any]]) -> str:
    """Formats retrieved vector search units into an attributed context block for the LLM."""
    if not raw_hits:
        return "No relevant meeting context retrieved."

    context_blocks = []
    for idx, hit in enumerate(raw_hits, 1):
        m_file = hit.get("filename") or hit.get("meeting_filename", f"Meeting #{hit.get('meeting_id')}")
        m_id = hit.get("meeting_id")
        doc_type = hit.get("doc_type", "transcript").replace("_", " ").title()
        score = hit.get("score", 0.0)
        text = hit.get("text", "")
        meta = hit.get("metadata", {})

        extra_info = []
        if meta.get("created_at"):
            extra_info.append(f"Date: {meta['created_at'][:10]}")
        if meta.get("deadline"):
            extra_info.append(f"Deadline: {meta['deadline']}")
        if meta.get("assignee") and meta["assignee"] != "Unassigned":
            extra_info.append(f"Assignee: {meta['assignee']}")
        if meta.get("priority"):
            extra_info.append(f"Priority: {meta['priority']}")

        extra_str = f" | {', '.join(extra_info)}" if extra_info else ""

        block = (
            f"[Source {idx}] Meeting: \"{m_file}\" (ID: {m_id})\n"
            f"Type: {doc_type}{extra_str} (Relevance Score: {score:.2f})\n"
            f"Content:\n{text.strip()}\n"
        )
        context_blocks.append(block)

    return "\n".join(context_blocks)

def generate_grounded_rag_answer(
    question: str,
    api_key: Optional[str] = None,
    meeting_id: Optional[int] = None,
    user_id: Optional[int] = None,
    doc_type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Task 3: RAG Search & AI Assistant
    Workflow:
    User Question -> Semantic Search -> Relevant Meetings -> RAG Context -> AI Answer
    Grounded strictly in retrieved context.
    """
    start_time = time.perf_counter()

    if not question or not question.strip():
        return {
            "question": question or "",
            "answer": "Please provide a question to search across the meeting repository.",
            "sources": [],
            "retrieval_time_seconds": 0.0,
            "total_time_seconds": 0.0,
            "is_grounded": False
        }

    search_res = semantic_search_meetings(
        query=question,
        top_k=top_k,
        doc_type=doc_type,
        meeting_id=meeting_id,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date
    )

    raw_hits = search_res.get("results", [])
    retrieval_time = search_res.get("search_time_seconds", 0.0)

    if not raw_hits:
        return {
            "question": question,
            "answer": "Based on the historical meetings in the knowledge base, I do not have enough information to answer this question.",
            "sources": [],
            "retrieval_time_seconds": retrieval_time,
            "total_time_seconds": round(time.perf_counter() - start_time, 4),
            "is_grounded": False
        }

    context_str = assemble_context(raw_hits)
    key_to_use = api_key or GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")

    full_prompt = RAG_SYSTEM_PROMPT.format(context=context_str) + f"\nUser Question: {question}\n\nDetailed Grounded Answer:"

    answer = ""
    if key_to_use:
        try:
            client = genai.Client(api_key=key_to_use)
            for model_name in ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-1.5-flash']:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=full_prompt,
                    )
                    if response and response.text:
                        answer = response.text.strip()
                        break
                except Exception as m_err:
                    logger.warning(f"RAG model '{model_name}' failed: {m_err}. Trying fallback model...")
        except Exception as e:
            logger.warning(f"Gemini client setup failed: {e}. Falling back to structured context extraction.")
            answer = ""

    # Factual structured fallback if Gemini is unreachable or offline
    if not answer:
        top_hit = raw_hits[0]
        meta = top_hit.get("metadata", {})
        dt = top_hit.get("doc_type", "").replace("_", " ").title()
        fn = top_hit.get("filename") or top_hit.get("meeting_filename", "")
        m_id = top_hit.get("meeting_id")
        created_at = meta.get("created_at", "")[:10] if meta.get("created_at") else "N/A"

        answer = (
            f"Based on the meeting **{fn}** (ID #{m_id}, Date: {created_at}), the most relevant {dt.lower()} states:\n\n"
            f"> \"{top_hit.get('text', '').strip()}\""
        )
        if meta.get("deadline"):
            answer += f"\n\n**Deadline:** {meta['deadline']}"
        if meta.get("assignee") and meta["assignee"] != "Unassigned":
            answer += f"\n**Assigned to:** {meta['assignee']}"

    # Build sources list
    sources = []
    for hit in raw_hits:
        meta = hit.get("metadata", {})
        created_at = meta.get("created_at", "")[:10] if meta.get("created_at") else None
        sources.append({
            "meeting_id": hit.get("meeting_id"),
            "filename": hit.get("filename") or hit.get("meeting_filename", ""),
            "doc_type": hit.get("doc_type", "transcript"),
            "preview": hit.get("text", "")[:180] + "...",
            "relevance_score": hit.get("score", 0.0),
            "date": created_at
        })

    elapsed = time.perf_counter() - start_time

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "retrieval_time_seconds": retrieval_time,
        "total_time_seconds": round(elapsed, 4),
        "is_grounded": True
    }

def stream_grounded_rag_answer(
    question: str,
    api_key: Optional[str] = None,
    meeting_id: Optional[int] = None,
    user_id: Optional[int] = None,
    doc_type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    top_k: int = 5
):
    """
    Generator yielding (token_chunk, sources, search_res) for Streamlit real-time streaming UI.
    Provides full M3 streaming interface with M4 multi-tenant user_id filtering.
    """
    search_res = semantic_search_meetings(
        query=question,
        top_k=top_k,
        doc_type=doc_type,
        meeting_id=meeting_id,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date
    )
    raw_hits = search_res.get("results", [])
    if not raw_hits:
        yield "Based on the historical meetings in the knowledge base, I do not have enough information to answer this question.", [], search_res
        return

    context_str = assemble_context(raw_hits)
    key_to_use = api_key or GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
    full_prompt = RAG_SYSTEM_PROMPT.format(context=context_str) + f"\nUser Question: {question}\n\nDetailed Grounded Answer:"

    sources = []
    for hit in raw_hits:
        meta = hit.get("metadata", {})
        created_at = meta.get("created_at", "")[:10] if meta.get("created_at") else None
        sources.append({
            "meeting_id": hit.get("meeting_id"),
            "filename": hit.get("filename") or hit.get("meeting_filename", ""),
            "doc_type": hit.get("doc_type", "transcript"),
            "preview": hit.get("text", "")[:180] + "...",
            "relevance_score": hit.get("score", 0.0),
            "date": created_at,
            "text": hit.get("text", "")
        })

    streamed = False
    if key_to_use:
        try:
            client = genai.Client(api_key=key_to_use)
            for model_name in ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-1.5-flash']:
                try:
                    response = client.models.generate_content_stream(
                        model=model_name,
                        contents=full_prompt,
                    )
                    for chunk in response:
                        if hasattr(chunk, 'text') and chunk.text:
                            yield chunk.text, sources, search_res
                            streamed = True
                    if streamed:
                        break
                except Exception:
                    continue
        except Exception:
            pass

    if not streamed:
        top_hit = raw_hits[0]
        meta = top_hit.get("metadata", {})
        dt = top_hit.get("doc_type", "").replace("_", " ").title()
        fn = top_hit.get("filename") or top_hit.get("meeting_filename", "")
        m_id = top_hit.get("meeting_id")
        created_at = meta.get("created_at", "")[:10] if meta.get("created_at") else "N/A"

        fallback = (
            f"Based on the meeting **{fn}** (ID #{m_id}, Date: {created_at}), the most relevant {dt.lower()} states:\n\n"
            f"> \"{top_hit.get('text', '').strip()}\""
        )
        if meta.get("deadline"):
            fallback += f"\n\n**Deadline:** {meta['deadline']}"
        if meta.get("assignee") and meta["assignee"] != "Unassigned":
            fallback += f"\n**Assigned to:** {meta['assignee']}"
        yield fallback, sources, search_res
