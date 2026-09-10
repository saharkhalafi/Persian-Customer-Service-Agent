from app.core.gemini import GeminiClient
from app.retrieval.knowledge_retriever import get_knowledge_retriever


class KnowledgeRepository:
    def __init__(self, gemini: GeminiClient):
        self.retriever = get_knowledge_retriever(gemini)

    def search(
        self,
        query: str,
        limit: int = 3,
    ) -> list[dict]:
        return self.retriever.search(
            query=query,
            limit=limit,
        )
