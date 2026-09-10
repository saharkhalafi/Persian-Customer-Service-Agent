from sqlalchemy import text
from sqlalchemy.orm import Session


class ConversationRepository:
    def __init__(self, db: Session):
        self.db = db
        self._ensure_table()

    def _ensure_table(self) -> None:
        self.db.execute(text("""
            CREATE TABLE IF NOT EXISTS public.conversation_messages (
                id BIGSERIAL PRIMARY KEY,
                customer_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """))
        self.db.execute(text("""
            CREATE INDEX IF NOT EXISTS
            idx_conversation_messages_customer_created
            ON public.conversation_messages (
                customer_id,
                created_at DESC
            )
        """))
        self.db.commit()

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
