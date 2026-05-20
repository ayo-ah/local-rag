import json
from pathlib import Path

import joblib
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from app.core.logging import get_logger
from app.utils.json_io import save_json

logger = get_logger(__name__)


class SectionClassifierTrainer:
    def __init__(
        self,
        dataset_path: Path,
        output_path: Path,
    ):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.eval_dir = self.output_path.parent / "evaluation" / self.output_path.stem 

    def load_dataset(self):
        logger.info(
            "section_dataset_loading_started",
            extra={"path": str(self.dataset_path)},
        )

        texts: list[str] = []
        labels: list[str] = []

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        for row in samples:
            texts.append(row["question"])
            labels.append(row["category"])

        logger.info(
            "section_dataset_loading_finished",
            extra={"samples": len(texts)},
        )

        return texts, labels
    
    def _wrap_labels(self, labels, width=20):
        return ['\n'.join(re.findall('.{1,' + str(width) + '}(?:\s+|$)', l)) for l in labels]

    def _save_evaluation(
        self,
        pipeline: Pipeline,
        y_val: list[str],
        y_pred: list[str],
    ) -> dict:

        self.eval_dir.mkdir(parents=True, exist_ok=True)

        report_dict = classification_report(
            y_val,
            y_pred,
            output_dict=True,
            zero_division=0,
        )

        save_json(
            self.eval_dir / "classification_report.json",
            report_dict,
        )

        labels = pipeline.classes_
        cm = confusion_matrix(y_val, y_pred, labels=labels)

        np.save(self.eval_dir / "confusion_matrix.npy", cm)

        wrapped_labels = self._wrap_labels(labels, width=20)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=wrapped_labels,
            yticklabels=wrapped_labels,
        )
        plt.ylabel("Фактические")
        plt.xlabel("Прогнозируемые")
        plt.title("Матрица ошибок")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.eval_dir / "confusion_matrix.png")
        plt.close()

        unique, counts = np.unique(y_val, return_counts=True)
        class_distribution = dict(zip(unique.tolist(), counts.tolist()))

        save_json(
            self.eval_dir / "class_distribution.json",
            class_distribution,
        )

        return {
            "classification_report": report_dict,
            "class_distribution": class_distribution,
        }
    def train(self) -> None:
        logger.info("section_classifier_training_started")

        texts, labels = self.load_dataset()

        X_train, X_val, y_train, y_val = train_test_split(
            texts,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels,
        )

        pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        max_features=20_000,
                        lowercase=True,
                        strip_accents="unicode",
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        )

        logger.info("section_classifier_fit_started")
        pipeline.fit(X_train, y_train)
        logger.info("section_classifier_fit_finished")

        y_pred = pipeline.predict(X_val)

        evaluation_metrics = self._save_evaluation(
            pipeline=pipeline,
            y_val=y_val,
            y_pred=y_pred,
        )

        logger.info(
            "section_classifier_validation_finished",
            extra={"macro_f1": evaluation_metrics["classification_report"]["macro avg"]["f1-score"]},
        )

        artifact = {
            "version": "2.1.0",
            "sklearn_version": sklearn.__version__,
            "pipeline": pipeline,
            "metrics": evaluation_metrics["classification_report"],
        }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, self.output_path)

        logger.info(
            "section_classifier_artifact_saved",
            extra={"path": str(self.output_path)},
        )

