from fastapi import Depends, Request

from app.db.session import get_db
from app.query.agents.query_agent import QueryAgent
from app.query.agents.factory import build_agent

async def get_query_agent(
    request: Request,
    db=Depends(get_db),
) -> QueryAgent:

    cfg = request.app.state.experiment_config

    return await build_agent(
        session=db,
        app_services=request.app.state.services,
        retrieval_config=cfg.retrieval,
        reranker_config=cfg.reranker,
        prompt_config=cfg.prompt,
    )
