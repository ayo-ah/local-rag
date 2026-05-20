from app.core.config.experiment_config import (
    RetrievalConfig,
    RerankerConfig,
    PromptConfig,
)
from app.query.agents.query_agent import QueryAgent
from app.query.analyzers.section import SectionAnalyzer
from app.classification.section.classifier import SectionClassifier
from app.retrieval.semantic import SemanticRetriever
from app.retrieval.lexical import LexicalRetriever
from app.retrieval.hybrid import HybridRetriever
from app.reranking.section_boost import SectionBoostReranker
from app.db.repositories.chunks import ChunkRepository
from app.db.repositories.embeddings import EmbeddingRepository
from app.core.services import AppServices
from app.prompting.builders.rag import RAGPromptBuilder

async def build_agent(
    session,
    app_services: AppServices,
    retrieval_config: RetrievalConfig,
    reranker_config: RerankerConfig,
    prompt_config: PromptConfig,
        ):

        prompt_builder = RAGPromptBuilder(
            prompt_root=prompt_config.root,
            task=prompt_config.task,
            version=prompt_config.version
        )

        chunk_repo = ChunkRepository(session)
        emb_repo = EmbeddingRepository(session)

        semantic = SemanticRetriever(
            embedding_service=app_services.embedding_service,
            embedding_repo=emb_repo,
            limit=retrieval_config.per_retriever_limits["semantic"]
        )

        if retrieval_config.retriever_type == "hybrid":
            lexical = LexicalRetriever(
                chunk_repo=chunk_repo,
                limit=retrieval_config.per_retriever_limits["lexical"],
            )
            retriever = HybridRetriever(
                [semantic, lexical],
                normalize_scores=True,
            )
        else:
            retriever = semantic

        ranker = None
        analyzers = []

        if reranker_config.reranker_type == "section_boost":
            classifier = SectionClassifier.load(
                path=reranker_config.section_classifier_path,
            )
            analyzers.append(SectionAnalyzer(
                 classifier=classifier,
                 threshold=reranker_config.section_threshold,
                 ))
            ranker = SectionBoostReranker(
                boost=reranker_config.reranker_boost
            )

        return QueryAgent(
            retriever=retriever,
            ranker=ranker,
            llm=app_services.llm_service,
            analyzers=analyzers,
            prompt_builder=prompt_builder
        )
