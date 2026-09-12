from app.core.context import RequestContext
from app.schemas.conversation_schemas import ConversationMessage
from app.services.conversation_service import ConversationService


class ConversationTools:
    def __init__(
        self,
        service: ConversationService,
    ):
        self.service = service

    def save_turn(
        self,
        context: RequestContext,
        user_message: str,
        assistant_message: str | None,
    ) -> None:
        self.service.save_turn(
            customer_id=context.customer_id,
            user_message=user_message,
            assistant_message=assistant_message,
        )

    def get_conversation_history(
        self,
        context: RequestContext,
        limit: int = 10,
    ) -> list[ConversationMessage]:
        return self.service.get_conversation_history(
            customer_id=context.customer_id,
            limit=limit,
        )

    def search_old_conversations(
        self,
        context: RequestContext,
        search_term: str,
        limit: int = 10,
    ) -> list[ConversationMessage]:
        return self.service.search_old_conversations(
            customer_id=context.customer_id,
            search_term=search_term,
            limit=limit,
        )
