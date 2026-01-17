from forecasting.logger.logging import get_logger
from forecasting.pipelines.inference import InferencePipeline
from pathlib import Path
from forecasting.utils.save_data import save_df_to_csv

model_path = Path("models/model_trained.pkl")
data_path = Path("data/inference.csv")
output_path = Path("data/predictions.csv")
plot_dir = Path("plots")

logger = get_logger()

try:
    logger.info("Starting Inference Script")
    inference = InferencePipeline(model_path=model_path)
    pred = inference.run(data_path=data_path, plot_dir=plot_dir, forecast_steps=12, save_plot=True)
    save_df_to_csv(output_path=output_path, df=pred)
    logger.debug(f"Predictions saved to {output_path} with {len(pred)} records")
except Exception:
    logger.exception("Failed to run inference script")
    raise