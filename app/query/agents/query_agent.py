from app.query.domain.dto import Query
from app.prompting.builders.base import PromptBuilder
from app.retrieval.base import BaseRetriever
from app.reranking.base import BaseReranker
from app.llm.base import LLMService
from app.query.analyzers.base import QueryAnalyzer
from app.query.domain.dto import QueryContext, QueryAgentResponse

class QueryAgent:
    def __init__(
        self,
        retriever: BaseRetriever,
        ranker: BaseReranker,
        llm: LLMService,
        analyzers: list[QueryAnalyzer],
        prompt_builder: PromptBuilder,
    ):
        self.retriever = retriever
        self.ranker = ranker
        self.llm = llm
        self.analyzers = analyzers
        self.prompt_builder = prompt_builder

    async def answer(self, query: Query) -> QueryAgentResponse:
        context = QueryContext(query=query)

        for analyzer in self.analyzers:
            await analyzer.analyze(context)   

        candidates = await self.retriever.retrieve(query=query.text)
        if self.ranker:
            top = await self.ranker.rerank(context=context, results=candidates, limit=query.top_k)
        else:
            top = candidates[:query.top_k]

        prompt = self.prompt_builder.build(
            context=context,
            context_chunks=[r.content for r in top],
        )
        generated_answer = await self.llm.generate(prompt)
        return QueryAgentResponse(answer=generated_answer, sources=top)

