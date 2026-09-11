from pydantic import BaseModel, Field
from typing import List, Optional

class ActionItemSchema(BaseModel):
    description: str = Field(description="The actual task or action item description.")
    assigned_participant: Optional[str] = Field(description="The name of the participant assigned to the action item. Use 'Unknown' if not explicitly stated.")
    deadline: Optional[str] = Field(description="The deadline or timeframe for the task, if specified.")
    priority: Optional[str] = Field(description="The priority of the task (e.g., High, Medium, Low), if inferable.")
    status: Optional[str] = Field(default="Pending", description="The status of the task. Usually 'Pending'.")

class MeetingIntelligenceSchema(BaseModel):
    summary: str = Field(description="A 1-2 paragraph executive summary of the meeting.")
    key_points: List[str] = Field(description="A list of key discussion points from the meeting.")
    decisions: List[str] = Field(description="A list of key decisions made during the meeting.")
    action_items: List[ActionItemSchema] = Field(description="A list of extracted action items.")
    participants: List[str] = Field(description="A list of all unique participant names identified in the meeting.")
