from pydantic import BaseModel


class KnowledgeChunk(BaseModel):
    chunk_id: int | None = None
    content: str
    page: int | None = None
    score: float | None = None
