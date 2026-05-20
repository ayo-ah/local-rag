import pytest
from uuid import uuid4
from pathlib import Path

from app.evaluation.dataset.generator import SyntheticDatasetGenerator
from app.prompting.builders.dataset_generation import DatasetPromptBuilder
from tests.mocks.mock_llm import MockLLMService
from tests.mocks.mock_chunk_repo import MockChunkRepository


@pytest.mark.asyncio
async def test_dataset_generation_pipeline():

    repo = MockChunkRepository()
    llm = MockLLMService()

    prompt_buider = DatasetPromptBuilder(
        Path("prompting/templates"),
        "dataset_generation",
        "qwen2.5_0.5b_v1.txt"
        )
    generator = SyntheticDatasetGenerator(
        chunks_repo=repo,
        llm=llm,
        questions_per_source=20,
        prompt_builder=prompt_buider
    )

    dataset = await generator.generate_for_source(
        source_id=uuid4(),
        limit_chunks=10,
    )

    # BASIC ASSERTS
    assert len(dataset) == 20

    # STRUCTURE CHECK
    for sample in dataset:

        assert "question" in sample
        assert "answer" in sample
        assert "type" in sample
        assert "source" in sample
        assert "score" in sample
        assert "reasoning" in sample

        assert isinstance(sample["question"], str)
        assert isinstance(sample["answer"], str)

    # TYPE DISTRIBUTION CHECK
    types = [s["type"] for s in dataset]

    assert "factual" in types
    assert "paraphrased" in types
    assert "cross_chunk" in types
    assert "negative" in types

    # NO EMPTY STRINGS
    for s in dataset:
        assert s["question"]
        assert s["answer"]
    
