from fastapi import APIRouter, Depends

from app.api.schemas.query import QueryRequest, QueryResponse
from app.api.dependencies import get_query_agent
from app.query.agents.query_agent import QueryAgent
from app.query.domain.dto import Query

router = APIRouter()


@router.post("/", response_model=QueryResponse)
async def ask(
    payload: QueryRequest,
    agent: QueryAgent = Depends(get_query_agent),
):
    query = Query(
        text=payload.question,
        top_k=payload.top_k,
    )

    agent_response = await agent.answer(query)

    return QueryResponse(
        answer=agent_response.answer,
        sources=[
            chunk.document_path
            for chunk in agent_response.sources
        ],
    )
