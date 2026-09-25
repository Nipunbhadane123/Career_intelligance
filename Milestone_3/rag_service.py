import os
import time
import logging
from typing import Dict, Any, Optional, List
from google import genai
from google.genai import types
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
        m_file = hit.get("meeting_filename", f"Meeting #{hit.get('meeting_id')}")
        m_id = hit.get("meeting_id")
        doc_type = hit.get("doc_type", "transcript").replace("_", " ").title()
        score = hit.get("score", 0.0)
        text = hit.get("text", "")
        meta = hit.get("metadata", {})

        extra_info = []
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
    doc_type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Task 5 Core Engine:
    End-to-End RAG Workflow:
    User Question -> Semantic Search -> Relevant Meeting Context -> LLM -> Grounded Answer
    """
    start_time = time.perf_counter()

    if not question or not question.strip():
        return {
            "question": question or "",
            "answer": "Please provide a valid question to search across the meeting knowledge base.",
            "sources": [],
            "retrieval_time_seconds": 0.0,
            "total_time_seconds": round(time.perf_counter() - start_time, 4),
            "is_grounded": False
        }

    # Step 1 & 2: Semantic Search across historical meetings
    search_res = semantic_search_meetings(
        query=question,
        top_k=top_k,
        doc_type=doc_type,
        meeting_id=meeting_id,
        start_date=start_date,
        end_date=end_date
    )

    raw_hits = search_res.get("raw_hits", [])
    if not raw_hits:
        return {
            "question": question,
            "answer": "I could not find any relevant meeting records matching your question in the knowledge repository.",
            "sources": [],
            "retrieval_time_seconds": search_res.get("retrieval_time_seconds", 0.0),
            "total_time_seconds": round(time.perf_counter() - start_time, 4),
            "is_grounded": False
        }

    # Step 3: Relevant Meeting Context Assembly
    context_str = assemble_context(raw_hits)

    # Step 4: LLM Generation
    key_to_use = api_key or os.getenv("GEMINI_API_KEY", "")
    full_prompt = RAG_SYSTEM_PROMPT.format(context=context_str) + f"\nUser Question: {question}\n\nDetailed Grounded Answer:"

    answer = ""
    if key_to_use:
        try:
            client = genai.Client(api_key=key_to_use)
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=full_prompt,
            )
            answer = response.text.strip()
        except Exception as e:
            logger.warning(f"Gemini API call failed: {e}. Falling back to structured context extraction.")
            answer = f"Notice: Live LLM synthesis unavailable ({e}). Showing retrieved grounded facts:\n\n" + "\n\n".join([f"• [{h.get('meeting_filename', 'Meeting')} - {h.get('doc_type', 'item')}]:\n  {h.get('text', '')}" for h in raw_hits[:3]])
    else:
        # Offline deterministic extraction when API key is not configured
        answer = (
            f"[Grounded Extraction from {len(raw_hits)} sources]:\n\n"
            + "\n\n".join([
                f"• From '{h.get('meeting_filename')}' ({h.get('doc_type')}):\n  {h.get('text')}"
                for h in raw_hits[:3]
            ])
        )

    total_time = time.perf_counter() - start_time

    return {
        "question": question,
        "answer": answer,
        "sources": raw_hits,
        "context_preview": context_str,
        "retrieval_time_seconds": search_res.get("retrieval_time_seconds", 0.0),
        "total_time_seconds": round(total_time, 4),
        "is_grounded": bool(raw_hits and len(raw_hits) > 0)
    }

def stream_grounded_rag_answer(
    question: str,
    api_key: str,
    meeting_id: Optional[int] = None,
    doc_type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    top_k: int = 5
):
    """
    Generator yielding (token_chunk, sources, search_res) for Streamlit real-time streaming UI.
    """
    search_res = semantic_search_meetings(
        query=question,
        top_k=top_k,
        doc_type=doc_type,
        meeting_id=meeting_id,
        start_date=start_date,
        end_date=end_date
    )
    raw_hits = search_res.get("raw_hits", [])

    if not raw_hits:
        yield "I could not find any relevant meeting records matching your question in the knowledge repository.", [], search_res
        return

    context_str = assemble_context(raw_hits)
    client = genai.Client(api_key=api_key)
    full_prompt = RAG_SYSTEM_PROMPT.format(context=context_str) + f"\nUser Question: {question}\n\nDetailed Grounded Answer:"

    try:
        response = client.models.generate_content_stream(
            model='gemini-2.5-flash',
            contents=full_prompt,
        )
        for chunk in response:
            if hasattr(chunk, 'text') and chunk.text:
                yield chunk.text, raw_hits, search_res
    except Exception as e:
        err_msg = str(e)
        if "API_KEY_HTTP_REFERRER_BLOCKED" in err_msg or "referer <empty>" in err_msg:
            fallback = (
                "⚠️ **Google Gemini API Key Restriction Detected**:\n\n"
                "Your Google Cloud API key has **'HTTP referrers (websites)'** restriction configured. "
                "Because Python backend calls send requests without browser referrers, Google blocked the live generation.\n\n"
                "👉 **Quick 1-Minute Fix**: In [Google Cloud Console Credentials](https://console.cloud.google.com/apis/credentials), click your API key and change **Application restrictions** to **'None'** (or generate an unrestricted server key from [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)).\n\n"
                "---\n### 🛡️ Grounded Facts Retrieved from Knowledge Store:\n\n"
            )
        else:
            fallback = f"*(Live Gemini generation unavailable: {e})*\n\n### 🛡️ Grounded Facts Retrieved from Knowledge Store:\n\n"
        fallback += "\n\n".join([f"• **[{h.get('meeting_filename', 'Meeting')} — {h.get('doc_type', 'item')}]**:\n  {h.get('text', '')}" for h in raw_hits[:3]])
        yield fallback, raw_hits, search_res
