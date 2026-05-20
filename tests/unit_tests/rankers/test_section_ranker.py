import uuid

import pytest

from app.reranking.section_boost import SectionBoostReranker
from app.retrieval.domain.dto import RetrievedChunk
from app.query.domain.dto import QueryContext


@pytest.mark.asyncio
async def test_section_ranker_boost_applied():
    ranker = SectionBoostReranker(boost=0.5)

    chunks = [
        RetrievedChunk(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            content="Про инфляцию",
            score=0.4,
            metadata={"section": "inflation"},
        ),
        RetrievedChunk(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            content="Про ключевую ставку",
            score=0.5,
            metadata={"section": "rates"},
        ),
    ]

    context = QueryContext(
        query="ключевая ставка",
        section_probs={
            "rates": 0.9,
            "inflation": 0.1,
        },
    )

    ranked = await ranker.rerank(
        results=chunks,
        context=context,
    )

    assert ranked[0].metadata["section"] == "rates"
    assert ranked[0].score > ranked[1].score
