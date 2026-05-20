import hashlib
import json
from typing import Any
from collections.abc import Mapping


def stable_json_dumps(data: Mapping[str, Any]) -> str:
    return json.dumps(
        data,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def hash_bytes(data: bytes, algorithm: str = "sha256") -> str:
    hasher = hashlib.new(algorithm)
    hasher.update(data)
    return hasher.hexdigest()


def hash_text(text: str, algorithm: str = "sha256") -> str:
    return hash_bytes(text.encode("utf-8"), algorithm)


def hash_document_content(
    content: str,
    metadata: Mapping[str, Any] | None = None,
    algorithm: str = "sha256",
) -> str:
    hasher = hashlib.new(algorithm)
    hasher.update(content.encode("utf-8"))

    if metadata:
        hasher.update(stable_json_dumps(metadata).encode("utf-8"))

    return hasher.hexdigest()


def hash_chunk(
    content: str,
    document_hash: str,
    chunk_index: int,
    algorithm: str = "sha256",
) -> str:
    hasher = hashlib.new(algorithm)
    hasher.update(document_hash.encode("utf-8"))
    hasher.update(str(chunk_index).encode("utf-8"))
    hasher.update(content.encode("utf-8"))
    return hasher.hexdigest()

def hash_config(
    config,
    algorithm: str = "sha256",
) -> str:
    if hasattr(config, "model_dump"):
        config = config.model_dump(
            mode="json",
            exclude_none=True,
        )
    json_text = stable_json_dumps(config)
    return hash_text(json_text, algorithm)[:8]
