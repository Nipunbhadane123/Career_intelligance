import re
import logging
from typing import Dict, Any, List, Optional
from database import get_meeting_by_id, get_user_meetings

logger = logging.getLogger(__name__)

def compute_meeting_analytics(meeting: Dict[str, Any]) -> Dict[str, Any]:
    """
    Task 2: Computes in-depth meeting analytics:
    - Duration & dialogue metrics
    - Action item distribution by priority & status
    - Participant contribution breakdown
    - Deadline distribution
    """
    transcript = meeting.get("transcript") or ""
    words = re.findall(r'\b\w+\b', transcript)
    word_count = len(words)

    # Participant speaking turns estimation from transcript (Speaker: "...")
    participant_turns = {}
    for p in meeting.get("participants", []):
        participant_turns[p] = 0

    lines = transcript.split("\n")
    for line in lines:
        for p in meeting.get("participants", []):
            if line.strip().startswith(p) or f"({p})" in line or f"{p}:" in line:
                participant_turns[p] = participant_turns.get(p, 0) + 1

    # Action items analysis
    action_items = meeting.get("action_items") or []
    by_priority = {"High": 0, "Medium": 0, "Low": 0}
    by_status = {"Pending": 0, "In Progress": 0, "Completed": 0}
    deadlines = []

    for ai in action_items:
        prio = ai.get("priority", "Medium")
        stat = ai.get("status", "Pending")
        by_priority[prio] = by_priority.get(prio, 0) + 1
        by_status[stat] = by_status.get(stat, 0) + 1
        if ai.get("deadline"):
            deadlines.append({
                "deadline": ai["deadline"],
                "description": ai.get("description"),
                "assignee": ai.get("assigned_participant")
            })

    # Sort deadlines
    deadlines.sort(key=lambda x: str(x["deadline"]))

    duration_sec = float(meeting.get("duration_seconds") or 0.0)
    if duration_sec == 0.0 and word_count > 0:
        # Estimate ~130 words per minute if duration is missing
        duration_sec = round((word_count / 130.0) * 60.0, 1)

    return {
        "meeting_id": meeting.get("id"),
        "filename": meeting.get("filename"),
        "duration_seconds": duration_sec,
        "duration_minutes": round(duration_sec / 60.0, 1),
        "word_count": word_count,
        "participant_count": len(meeting.get("participants", [])),
        "participant_contributions": participant_turns,
        "action_items_total": len(action_items),
        "action_items_by_priority": by_priority,
        "action_items_by_status": by_status,
        "decisions_count": len(meeting.get("key_decisions", [])),
        "deadlines": deadlines
    }

def compute_user_overview_analytics(user_id: Optional[int] = None) -> Dict[str, Any]:
    """Compute aggregate analytics across meetings."""
    meetings = get_user_meetings(user_id=user_id)
    total_meetings = len(meetings)
    total_duration_sec = 0.0
    total_actions = 0
    completed_actions = 0
    pending_actions = 0
    total_decisions = 0
    all_participants = set()
    platform_counts = {"upload": 0, "zoom": 0, "google_meet": 0}

    for m in meetings:
        total_duration_sec += float(m.get("duration_seconds") or 0.0)
        p = m.get("platform") or "upload"
        platform_counts[p] = platform_counts.get(p, 0) + 1

        ais = m.get("action_items") or []
        total_actions += len(ais)
        for a in ais:
            if a.get("status") == "Completed":
                completed_actions += 1
            else:
                pending_actions += 1

        total_decisions += len(m.get("key_decisions") or [])
        for part in m.get("participants") or []:
            all_participants.add(part)

    completion_rate = (completed_actions / total_actions * 100.0) if total_actions > 0 else 0.0

    return {
        "total_meetings": total_meetings,
        "total_duration_hours": round(total_duration_sec / 3600.0, 2),
        "total_action_items": total_actions,
        "completed_action_items": completed_actions,
        "pending_action_items": pending_actions,
        "completion_rate_percent": round(completion_rate, 1),
        "total_decisions": total_decisions,
        "total_participants": len(all_participants),
        "unique_participants_count": len(all_participants),
        "platform_breakdown": platform_counts
    }
