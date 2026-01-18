from forecasting.pipelines.train_pipeline import ModelTrainer
from forecasting.logger.logging import get_logger
from pathlib import Path
from lightgbm import LGBMRegressor

logger = get_logger()
data_path = Path("data/Walmart_final.csv")
save_model_path = Path("models/model_trained.pkl")

trainer = ModelTrainer(data_path=data_path)

metrics = trainer.run(model=LGBMRegressor, save_model_path=save_model_path)
