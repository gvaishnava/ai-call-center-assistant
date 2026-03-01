from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
from datetime import datetime

class SentimentEnum(str, Enum):
    POSITIVE = "Positive"
    NEUTRAL = "Neutral"
    NEGATIVE = "Negative"

class CallMetadata(BaseModel):
    call_id: str = Field(..., description="Unique identifier for the call")
    customer_name: Optional[str] = Field(None, description="Name of the customer")
    agent_name: Optional[str] = Field(None, description="Name of the call center agent")
    timestamp: datetime = Field(default_factory=datetime.now, description="Time of the call")

class TranscriptionResult(BaseModel):
    text: str = Field(..., description="Full text transcript of the call")
    language: str = Field("en", description="Detected language of the call")

class CallSummary(BaseModel):
    one_line_summary: str = Field(..., description="A single sentence summarizing the call")
    key_points: List[str] = Field(..., description="List of main discussion points")
    sentiment: SentimentEnum = Field(..., description="Overall sentiment of the call")
    action_items: List[str] = Field(..., description="Tasks or follow-ups identified in the call")

class QualityScore(BaseModel):
    technical_score: int = Field(..., ge=1, le=10, description="Technical knowledge and issue resolution score, based on rubric")
    professionalism_score: int = Field(..., ge=1, le=10, description="Professionalism and demeanor score, based on rubric")
    communication_score: int = Field(..., ge=1, le=10, description="Communication clarity and effectiveness score, based on rubric")
    process_adherence_score: int = Field(..., ge=1, le=10, description="Process and policy adherence score, based on rubric")
    soft_skills_score: int = Field(..., ge=1, le=10, description="Soft skills and empathy score, based on rubric")
    rubric_notes: str = Field(..., description="Detailed notes justifying the scores based explicitly on the provided rubric brackets")
    customer_sentiment_overall: SentimentEnum = Field(..., description="Overall sentiment of the customer during the call")
    customer_primary_emotion: str = Field(..., description="Primary emotion expressed by the customer (e.g., Frustrated, Angry, Happy, Neutral)")
    agent_tone: str = Field(..., description="The tone of the agent during the conversation (e.g., Empathetic, Professional, Impatient)")
    sentiment_shift: str = Field(..., description="How the sentiment changed from the beginning to the end of the call (e.g., 'Negative to Neutral')")
    churn_risk_detected: bool = Field(..., description="Whether the customer exhibited signs of churning or canceling their service")
