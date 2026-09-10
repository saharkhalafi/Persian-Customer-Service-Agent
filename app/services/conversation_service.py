from app.repositories.conversation_repository import ConversationRepository
from app.schemas.conversation_schemas import ConversationMessage


ALLOWED_ROLES = {"user", "assistant"}


class ConversationService:
    def __init__(
        self,
        repository: ConversationRepository,
    ):
        self.repository = repository

    def save_turn(
        self,
        customer_id: str,
        user_message: str,
        assistant_message: str | None,
    ) -> None:
        user_message = (user_message or "").strip()[:4000]

        if user_message:
            self.repository.add_message(
                customer_id=customer_id,
                role="user",
                content=user_message,
            )

        assistant_message = (assistant_message or "").strip()[:4000]

        if assistant_message:
            self.repository.add_message(
                customer_id=customer_id,
                role="assistant",
                content=assistant_message,
            )

    def get_conversation_history(
        self,
        customer_id: str,
        limit: int = 10,
    ) -> list[ConversationMessage]:
        limit = min(max(limit, 1), 50)

        rows = self.repository.get_recent_messages(
            customer_id=customer_id,
            limit=limit,
        )

        return self._to_messages(rows)

    def search_old_conversations(
        self,
        customer_id: str,
        search_term: str,
        limit: int = 10,
    ) -> list[ConversationMessage]:
        search_term = search_term.strip()[:200]

        if not search_term:
            return []

        limit = min(max(limit, 1), 50)

        rows = self.repository.search_messages(
            customer_id=customer_id,
            search_term=search_term,
            limit=limit,
        )

        return self._to_messages(rows)

    def _to_messages(
        self,
        rows: list[dict],
    ) -> list[ConversationMessage]:
        messages = []

        for row in rows:
            role = str(row.get("role") or "").strip()
            content = str(row.get("content") or "").strip()

            if role not in ALLOWED_ROLES or not content:
                continue

            messages.append(
                ConversationMessage(
                    role=role,
                    content=content,
                    created_at=row.get("created_at"),
                )
            )

        return messages
