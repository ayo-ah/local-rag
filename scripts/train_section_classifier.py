import argparse
from pathlib import Path
from app.classification.section.trainer import SectionClassifierTrainer
from app.core.config.env_settings import get_log_settings
from app.core.logging import setup_logging

def main(args):
    
    setup_logging(get_log_settings().log_level)

    trainer = SectionClassifierTrainer(
        dataset_path=Path(args.dataset_path),
        output_path=Path(args.output_path),
        )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="data/experiments/references/section/train_lr_saiga.json")
    parser.add_argument("--output_path", default="app/classification/artifacts/section_classifier_v1.joblib")
    args = parser.parse_args()
    main(args)
