# models.py

from pydantic import BaseModel, Field
from typing import List, Optional

class InterviewTranscript(BaseModel):
    id: int
    length_of_time_spoken_seconds: float = Field(..., ge=0)
    text_context: str = Field(..., min_length=1)
    speaker_name: str = Field(..., min_length=1)

class NewsArticle(BaseModel):
    id: int
    title: str = Field(..., min_length=1)
    author: str = Field(..., min_length=1)
    publication_date: str = Field(..., regex=r"^\d{4}-\d{2}-\d{2}$")  # Format YYYY-MM-DD
    content: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    tags: List[str]
    url: str = Field(..., min_length=1, regex=r"^(http|https)://[^ ]+$")  # Basic URL validation
