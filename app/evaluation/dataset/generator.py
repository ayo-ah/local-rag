import json
import random
import uuid
from pathlib import Path
from typing import Any

from app.db.repositories.chunks import ChunkRepository
from app.llm.base import LLMService
from app.evaluation.domain.dto import EvalChunk, EvalSample
from app.prompting.builders.dataset_generation import DatasetPromptBuilder
from app.utils.json_io import append_jsonl
from app.core.logging import get_logger


logger = get_logger(__name__)


REQUIRED_FIELDS = {"question", "answer", "score", "reasoning"}
MIN_SCORE = 1.0
MAX_SCORE = 5.0


class SyntheticDatasetGenerator:

    def __init__(
        self,
        *,
        chunks_repo: ChunkRepository,
        llm: LLMService,
        prompt_builder: DatasetPromptBuilder,
        questions_per_source: int = 20,
        raw_output_path: str = "llm_raw_outputs.jsonl",
    ):
        self._chunks_repo = chunks_repo
        self._llm = llm
        self._prompt_builder = prompt_builder
        self._questions_per_source = questions_per_source
        self._raw_output_path = Path(raw_output_path)

    async def generate_for_source(
        self,
        *,
        source_id,
        limit_chunks: int = 50,
    ) -> list[EvalSample]:

        chunks = await self._chunks_repo.get_chunks_for_evaluation(
            source_id=source_id,
            limit=limit_chunks,
        )

        if len(chunks) < 3:
            raise ValueError("Not enough chunks to generate dataset")

        samples = await self._generate_factual_samples(
            chunks=chunks,
            count=self._questions_per_source,
            source_id=source_id,
        )

        random.shuffle(samples)
        logger.info("generated_dataset_size=%s", len(samples))
        return samples

    async def _generate_factual_samples(
        self,
        *,
        chunks: list[EvalChunk],
        count: int,
        source_id,
    ) -> list[EvalSample]:

        selected = random.sample(chunks, min(count, len(chunks)))
        results: list[EvalSample] = []

        for chunk in selected:
            sample = await self._process_chunk(
                chunk=chunk,
                source_id=source_id,
            )
            if sample:
                results.append(sample)

        return results

    async def _process_chunk(
        self,
        *,
        chunk: EvalChunk,
        source_id,
    ) -> EvalSample | None:

        prompt = self._prompt_builder.build(chunk.content)
        parsed, raw_output = await self._ask_llm(prompt)

        append_jsonl(self._raw_output_path, {
            "source_id": str(source_id),
            "document_path": chunk.document_path,
            "raw_output": raw_output,
        })

        if not parsed:
            return None

        validated = self._validate_payload(parsed)
        if not validated:
            return None

        try:
            return self._map_to_sample(validated, chunk)
        except Exception as e:
            logger.exception(
                "sample_mapping_failed chunk=%s error=%s",
                chunk.document_path,
                str(e),
            )
            return None

    async def _ask_llm(
        self,
        prompt: str,
        max_retries: int = 1,
    ) -> tuple[dict[str, Any] | None, str]:

        last_raw = ""

        for _ in range(max_retries):
            raw = await self._llm.generate(prompt)
            last_raw = raw

            parsed = self._parse_json(raw)
            if parsed is not None:
                return parsed, raw

            logger.warning("invalid_json_from_llm output=%s", raw)

        return None, last_raw

    def _parse_json(self, text: str) -> dict[str, Any] | None:
        text = text.strip()
        start = text.find("{")
        if start == -1:
            return None

        brace_count = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                brace_count += 1
            elif text[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        return None
        return None

    def _validate_payload(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any] | None:

        if not isinstance(payload, dict):
            return None

        if not REQUIRED_FIELDS.issubset(payload.keys()):
            return None

        try:
            score = float(payload["score"])
        except (TypeError, ValueError):
            return None

        if not MIN_SCORE <= score <= MAX_SCORE:
            return None

        payload["score"] = score
        return payload

    def _map_to_sample(
        self,
        payload: dict[str, Any],
        chunk: EvalChunk,
    ) -> EvalSample:

        return EvalSample(
            question_id=str(uuid.uuid4()),
            question=payload["question"],
            question_type="factual",
            reference_answer=payload["answer"],
            score=payload["score"],
            reasoning=payload["reasoning"],
            source=[chunk.document_path],
        )
