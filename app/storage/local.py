import os
import aiofiles
from pathlib import Path

from app.storage.base import BaseStorage
from app.core.logging import get_logger


logger = get_logger(__name__)


class LocalStorage(BaseStorage):
    def __init__(self, root_path: str):
        self.root_path = os.path.abspath(root_path)

    def _resolve_path(self, reference: str) -> str:
        if os.path.isabs(reference):
            return reference
        return os.path.join(self.root_path, reference)

    async def load(self, reference: str) -> bytes:
        path = self._resolve_path(reference)

        if not os.path.exists(path):
            logger.warning(
                "File_not_found_in_local_storage",
                extra={"reference": reference},
            )
            raise FileNotFoundError(path)

        try:
            async with aiofiles.open(path, "rb") as f:
                return await f.read()
        except Exception:
            logger.exception(
                "Failed_to_read_file_from_local_storage",
                extra={"reference": reference},
            )
            raise

    async def exists(self, reference: str) -> bool:
        path = self._resolve_path(reference)
        return os.path.exists(path)

    async def save(self, reference: str, data: bytes) -> str:
        path = self._resolve_path(reference)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        try:
            async with aiofiles.open(path, "wb") as f:
                await f.write(data)
        except Exception:
            logger.exception(
                "Failed_to_write_file_to_local_storage",
                extra={"reference": reference},
            )
            raise

        return reference
    
    async def list_files(self, pattern: str = "**/*") -> list[str]:
        files = []
        for file_path in Path(self.root_path).rglob(pattern):
            if file_path.is_file() and not file_path.name.startswith('.'):
                reference = str(file_path.relative_to(self.root_path))
                files.append(reference)
        return files
    