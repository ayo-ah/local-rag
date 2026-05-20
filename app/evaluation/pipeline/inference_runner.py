from app.query.domain.dto import Query
from ..domain.dto import RAGResponse
from app.query.agents.query_agent import QueryAgent, QueryAgentResponse
from app.evaluation.artifacts import write_inference_artifact
from app.evaluation.dataset.loader import load_dataset
from app.core.config.experiment_config import ExperimentConfig
async def run_inference(
        agent: QueryAgent, 
        experiment_config: ExperimentConfig,
        ) -> list[RAGResponse]:
    
    results = []
    samples = load_dataset(experiment_config.dataset_path)

    for sample in samples:
        query = Query(
            text=sample.question, 
            top_k=experiment_config.retrieval.top_k
            )
        response: QueryAgentResponse = await agent.answer(query)

        results.append(
            RAGResponse(
                question_id=sample.question_id,
                question=sample.question,
                answer=response.answer,
                sources=[chunk.document_path for chunk in response.sources],
            )
        )
    
    write_inference_artifact(
        responses=results,
        config=experiment_config,
    )
    return results
