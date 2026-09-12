from app.core.gemini import GeminiClient
from app.core.observability import traced
from app.retrieval.knowledge_retriever import get_knowledge_retriever


class KnowledgeRepository:
    def __init__(self, gemini: GeminiClient):
        self.retriever = get_knowledge_retriever(gemini)

    @traced("db_operation", layer="knowledge_repository", operation="search")
    def search(
        self,
        query: str,
        limit: int = 3,
    ) -> list[dict]:
        return self.retriever.search(
            query=query,
            limit=limit,
        )
