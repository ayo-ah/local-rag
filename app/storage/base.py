from typing import Protocol

class BaseStorage(Protocol):
    root_path: str
    
    async def load(self, reference: str) -> bytes:
        ...

    async def exists(self, reference: str) -> bool:
        ...

    async def save(self, reference: str, data: bytes) -> str:
        ...
    
    async def list_files(self, pattern: str = "**/*") -> list[str]:
        ...
