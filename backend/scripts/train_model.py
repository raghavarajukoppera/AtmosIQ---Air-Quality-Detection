from __future__ import annotations

import json

from app.ml.trainer import GCNTrainer
from app.services.logging_utils import get_logger
from app.services.metro_pipeline import MetroPipeline


def main() -> None:
    logger = get_logger("train_script")
    pipeline = MetroPipeline()
    prepared = pipeline.prepare_training_data(force_refresh=True)

    logger.info("Dataset summary: %s", json.dumps(prepared.summary, indent=2))
    trainer = GCNTrainer()
    result = trainer.train(
        samples=prepared.samples,
        graphs=prepared.graphs,
        station_df=prepared.station_df,
        training_summary=prepared.summary,
        max_epochs=40,
    )
    logger.info(
        "Training complete | train=%s val=%s test=%s | MAE=%.4f RMSE=%.4f | model=%s",
        result.train_samples,
        result.val_samples,
        result.test_samples,
        result.test_mae,
        result.test_rmse,
        result.model_path,
    )


if __name__ == "__main__":
    main()
