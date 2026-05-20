from app.query.analyzers.base import QueryAnalyzer
from app.query.domain.dto import QueryContext
from app.classification.section.classifier import SectionClassifier
from app.core.logging import get_logger


logger = get_logger(__name__)


class SectionAnalyzer(QueryAnalyzer):
    def __init__(self, classifier: SectionClassifier, threshold: float = 0.6,):
        self.classifier = classifier
        self.threshold = threshold

    async def analyze(self, context: QueryContext) -> None:
        try:
            probs = self.classifier.predict_proba(context.query.text)
        except Exception:
            logger.exception("Section_classification_failed")
            return
        
        if not probs:
            return
        
        best_section, best_prob = max(
            probs.items(),
            key=lambda x: x[1],
        )
        context.features["section_confidence"] = best_prob 

        if best_prob < self.threshold or best_section == "Другое" :
            logger.info(
                "section_confidence_below_threshold",
                extra={
                    "confidence": best_prob,
                    "threshold": self.threshold,
                },
            )
            return


        context.section_probs = probs

        logger.debug(
            "Section_predicted",
            extra={
                "section": best_section,
                "confidence": round(best_prob, 4),
            },
        )
