import json
import logging
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from schemas import MeetingIntelligenceSchema

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert AI meeting assistant. Analyze the following meeting transcript and extract structured meeting intelligence.

You must identify:
1. A concise executive summary of the meeting.
2. The key discussion points.
3. Any key decisions made.
4. Action items, including their description, the assigned participant (or 'Unknown'), priority (High, Medium, Low), and deadline if mentioned.
5. A list of all unique participants involved in the meeting.

Transcript:
"""

def is_retryable_error(exception: BaseException) -> bool:
    err_str = str(exception)
    if "403" in err_str or "PERMISSION_DENIED" in err_str or "API_KEY_HTTP_REFERRER_BLOCKED" in err_str:
        return False
    return True

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception(is_retryable_error),
    reraise=True
)
def process_transcript_with_llm(client: genai.Client, transcript: str) -> MeetingIntelligenceSchema:
    """
    Processes the transcript using the Gemini LLM with structured output validation.
    Retries up to 3 times on failure.
    """
    prompt = SYSTEM_PROMPT + f"\n{transcript}"
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=MeetingIntelligenceSchema,
        ),
    )
    
    try:
        response_text = response.text
        if response_text.startswith("```json"):
            response_text = response_text[7:-3].strip()
        elif response_text.startswith("```"):
            response_text = response_text[3:-3].strip()
            
        data = json.loads(response_text)
        return MeetingIntelligenceSchema(**data)
    except Exception as e:
        logger.error(f"Failed to parse LLM structured output: {e}\nRaw output: {response.text}")
        raise e
