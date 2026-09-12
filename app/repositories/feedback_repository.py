from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.observability import traced


class FeedbackRepository:
    def __init__(self, db: Session):
        self.db = db
        self._ensure_table()

    def _ensure_table(self) -> None:
        self.db.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS public.message_feedback (
                    id BIGSERIAL PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    conversation_id TEXT,
                    message_id TEXT,
                    rating TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
        )
        self.db.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_message_feedback_customer_created
                ON public.message_feedback (customer_id, created_at DESC)
                """
            )
        )
        self.db.commit()

    @traced("db_operation", layer="feedback_repository", operation="add_feedback")
    def add(
        self,
        customer_id: str,
        rating: str,
        conversation_id: str | None = None,
        message_id: str | None = None,
    ) -> None:
        self.db.execute(
            text(
                """
                INSERT INTO public.message_feedback (
                    customer_id,
                    conversation_id,
                    message_id,
                    rating
                )
                VALUES (
                    :customer_id,
                    :conversation_id,
                    :message_id,
                    :rating
                )
                """
            ),
            {
                "customer_id": customer_id,
                "conversation_id": conversation_id or None,
                "message_id": message_id or None,
                "rating": rating,
            },
        )
        self.db.commit()
