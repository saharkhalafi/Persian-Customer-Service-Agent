from app.repositories.knowledge_repository import KnowledgeRepository
from app.schemas.knowledge_schemas import KnowledgeChunk


class KnowledgeService:
    def __init__(
        self,
        repository: KnowledgeRepository,
    ):
        self.repository = repository

    def search_knowledge_base(
        self,
        query: str,
        limit: int = 3,
    ) -> list[KnowledgeChunk]:
        query = query.strip()

        if not query:
            return []

        query = query[:500]
        limit = min(max(limit, 1), 10)

        rows = self.repository.search(
            query=query,
            limit=limit,
        )

        return [
            KnowledgeChunk(
                chunk_id=row.get("chunk_id"),
                content=row.get("content") or "",
                page=row.get("page"),
                score=row.get("score"),
            )
            for row in rows
            if row.get("content")
        ]
