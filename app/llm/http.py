import asyncio
import json
import httpx

from app.llm.base import LLMService
from app.core.logging import get_logger


logger = get_logger(__name__)


class LlamaCppHttpLLMService(LLMService):
    def __init__(
        self,
        base_url: str,
        timeout: int = 1000,
        max_tokens: int = 256,
        temperature: float = 0.1,
        stop: list[str] | None = None,
        max_retries: int = 10,
    ):
        self.base_url = base_url
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.stop = stop or []
        self.max_retries = max_retries

        self._create_client()

    def _create_client(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout),
        )

    async def _recreate_client(self) -> None:
        try:
            await self._client.aclose()
        except Exception:
            pass
        self._create_client()

    async def generate(self, prompt: str) -> str:
        attempt = 0

        while True:
            try:
                async with self._client.stream(
                    "POST",
                    "/completion",
                    json={
                        "prompt": prompt,
                        "n_predict": self.max_tokens,
                        "temperature": self.temperature,
                        "stop": self.stop,
                        "top_p": 0.8,
                        "top_k": 20,
                        "min_p": 0,
                        "presence_penalty": 1.5,
                        "repeat_penalty": 1.1,
                        "stream": True,
                    },
                ) as response:

                    response.raise_for_status()

                    output = ""

                    async for line in response.aiter_lines():

                        if not line:
                            continue

                        data = line.strip()

                        if data.startswith("data:"):
                            data = data[5:].strip()

                        if data == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        token = chunk.get("content")
                        if token:
                            output += token

                    return output

            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code

                if 400 <= status_code < 500:
                    logger.error(
                        "LLM_client_error",
                        extra={
                            "status_code": status_code,
                            "response_text": e.response.text,
                        },
                    )
                    raise

                attempt += 1
                if attempt > self.max_retries:
                    logger.error(
                        "LLM_server_error_max_retries_exceeded",
                        extra={"status_code": status_code},
                    )
                    raise

                wait_time = min(2 ** attempt, 30)

                logger.warning(
                    "LLM_server_error_retrying",
                    extra={
                        "status_code": status_code,
                        "attempt": attempt,
                        "wait_time": wait_time,
                    },
                )

                await asyncio.sleep(wait_time)
                await self._recreate_client()

            except httpx.RequestError as e:
                attempt += 1

                if attempt > self.max_retries:
                    logger.error(
                        "LLM_network_error_max_retries_exceeded",
                        extra={"error": str(e)},
                    )
                    raise

                wait_time = min(2 ** attempt, 30)

                logger.warning(
                    "LLM_network_error_retrying",
                    extra={
                        "error": str(e),
                        "attempt": attempt,
                        "wait_time": wait_time,
                    },
                )

                await asyncio.sleep(wait_time)
                await self._recreate_client()