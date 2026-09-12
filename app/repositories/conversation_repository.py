from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.observability import traced


class ConversationRepository:
    def __init__(self, db: Session):
        self.db = db

    @traced("db_operation", layer="conversation_repository", operation="add_message")
    def add_message(
        self,
        customer_id: str,
        role: str,
        content: str,
    ) -> None:
        self.db.execute(
            text("""
                INSERT INTO public.conversation_messages (
                    customer_id,
                    role,
                    content
                )
                VALUES (
                    :customer_id,
                    :role,
                    :content
                )
            """),
            {
                "customer_id": customer_id,
                "role": role,
                "content": content,
            },
        )
        self.db.commit()

    @traced("db_operation", layer="conversation_repository", operation="get_recent_messages")
    def get_recent_messages(
        self,
        customer_id: str,
        limit: int = 10,
    ) -> list[dict]:
        limit = min(max(limit, 1), 50)

        rows = self.db.execute(
            text("""
                SELECT
                    role,
                    content,
                    created_at::text AS created_at
                FROM public.conversation_messages
                WHERE customer_id = :customer_id
                ORDER BY created_at DESC, id DESC
                LIMIT :limit
            """),
            {
                "customer_id": customer_id,
                "limit": limit,
            },
        ).mappings()

        messages = [dict(row) for row in rows]
        messages.reverse()
        return messages

    @traced("db_operation", layer="conversation_repository", operation="search_messages")
    def search_messages(
        self,
        customer_id: str,
        search_term: str,
        limit: int = 10,
    ) -> list[dict]:
        limit = min(max(limit, 1), 50)

        rows = self.db.execute(
            text("""
                SELECT
                    role,
                    content,
                    created_at::text AS created_at
                FROM public.conversation_messages
                WHERE
                    customer_id = :customer_id
                    AND content ILIKE :term
                ORDER BY created_at DESC, id DESC
                LIMIT :limit
            """),
            {
                "customer_id": customer_id,
                "term": f"%{search_term}%",
                "limit": limit,
            },
        ).mappings()

        return [dict(row) for row in rows]
