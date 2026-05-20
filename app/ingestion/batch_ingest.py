import time

from app.ingestion.ingest_service import IngestService
from app.storage.base import BaseStorage
from app.db.repositories.sources import SourceRepository
from app.core.logging import get_logger


logger = get_logger(__name__)


class BatchIngestService:
    def __init__(
        self,
        storage: BaseStorage,
        ingest_service: IngestService,
        source_repo: SourceRepository,
    ):
        self.storage = storage
        self.ingest_service = ingest_service
        self.source_repo = source_repo


    async def ingest_batch(
        self,
        *,
        source_type: str,
        source_name: str,
        mime_detector,
    ) -> dict:
        refs = await self.storage.list_files()
        total = len(refs)

        batch_start_time = time.perf_counter()
        processed_times: list[float] = []

        logger.info(
            "Batch_ingestion_started",
            extra={
                "source_type": source_type,
                "source_name": source_name,
                "references_count": total,
            },
        )

        source = await self.source_repo.get_or_create(
            type=source_type,
            name=source_name,
            config={"root": self.storage.root_path},
        )

        results = {"success": [], "failed": []}

        for index, ref in enumerate(refs, start=1):
            doc_start_time = time.perf_counter()

            logger.info(
                "Ingestion_started_for_reference",
                extra={
                    "reference": ref,
                    "document_index": index,
                    "documents_total": total,
                },
            )

            try:
                data = await self.storage.load(ref)
                mime_type = mime_detector(ref)

                document = await self.ingest_service.ingest(
                    source_id=source.id,
                    reference=ref,
                    data=data,
                    mime_type=mime_type,
                    metadata={},
                )

                results["success"].append(document.id)

                status = "success"

            except Exception as e:
                logger.warning(
                    "Ingestion_failed_for_reference",
                    extra={
                        "reference": ref,
                        "document_index": index,
                        "documents_total": total,
                        "error": str(e),
                    },
                )

                results["failed"].append(
                    {"reference": ref, "error": str(e)}
                )

                status = "failed"

            doc_duration = time.perf_counter() - doc_start_time
            processed_times.append(doc_duration)

            avg_time = sum(processed_times) / len(processed_times)
            remaining_docs = total - index
            eta_seconds = remaining_docs * avg_time

            logger.info(
                "Ingestion_finished_for_reference",
                extra={
                    "reference": ref,
                    "document_index": index,
                    "documents_total": total,
                    "status": status,
                    "document_processing_time_sec": round(doc_duration, 3),
                    "average_time_per_doc_sec": round(avg_time, 3),
                    "eta_seconds": round(eta_seconds, 1),
                    "progress_percent": round(index / total * 100, 2),
                },
            )

        batch_duration = time.perf_counter() - batch_start_time

        logger.info(
            "Batch_ingestion_completed",
            extra={
                "source_type": source_type,
                "source_name": source_name,
                "success_count": len(results["success"]),
                "failed_count": len(results["failed"]),
                "batch_duration_sec": round(batch_duration, 2),
                "avg_time_per_doc_sec": round(
                    (batch_duration / total) if total else 0, 3
                ),
            },
        )

        return results
