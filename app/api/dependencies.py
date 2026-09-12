from functools import lru_cache

from fastapi import Depends
from sqlalchemy.orm import Session

from app.agents.agent import CustomerSupportAgent
from app.core.container import build_agent
from app.core.database import get_db
from app.core.gemini import GeminiClient
from app.repositories.feedback_repository import FeedbackRepository


@lru_cache
def get_gemini_client() -> GeminiClient:
    return GeminiClient()


def get_agent(
    db: Session = Depends(get_db),
    gemini: GeminiClient = Depends(get_gemini_client),
) -> CustomerSupportAgent:

    return build_agent(
        db=db,
        gemini=gemini,
    )


def get_feedback_repository(
    db: Session = Depends(get_db),
) -> FeedbackRepository:
    return FeedbackRepository(db)