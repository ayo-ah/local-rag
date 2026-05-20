from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(default=5, ge=1, le=50)


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]