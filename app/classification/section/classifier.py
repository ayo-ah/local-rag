import joblib
from typing import Tuple
from pathlib import Path

from app.core.logging import get_logger


logger = get_logger(__name__)


class SectionClassifier:
    def __init__(
        self,
        pipeline,
    ):
        self.pipeline = pipeline

    @classmethod
    def load(cls, path: Path):
        logger.info(
            "loading_section_classifier",
            extra={"path": str(path)},
        )

        payload = joblib.load(path)

        logger.info(
            "section_classifier_loaded",
            extra={
                "version": payload.get("version"),
                "sklearn_version": payload.get("sklearn_version"),
            },
        )

        return cls(
            pipeline=payload["pipeline"],
        )

    def predict_proba(self, query: str) -> dict[str, float]:
        try:
            probs = self.pipeline.predict_proba([query])[0]
            labels = self.pipeline.classes_

        except Exception:
            logger.exception(
                "section_classifier_prediction_failed",
                extra={"query": query},
            )
            raise

        return dict(zip(labels, probs.astype(float)))

    def predict(self, query: str) -> Tuple[str, float]:
        probs = self.predict_proba(query)

        label, score = max(probs.items(), key=lambda x: x[1])

        return label, score