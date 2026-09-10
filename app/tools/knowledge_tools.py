from app.core.context import RequestContext
from app.schemas.knowledge_schemas import KnowledgeChunk
from app.services.knowledge_service import KnowledgeService


class KnowledgeTools:
    def __init__(
        self,
        service: KnowledgeService,
    ):
        self.service = service

    def search_knowledge_base(
        self,
        context: RequestContext,
        query: str,
        limit: int = 3,
    ) -> list[KnowledgeChunk]:
        _ = context
        return self.service.search_knowledge_base(
            query=query,
            limit=limit,
        )
