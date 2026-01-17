from src.forecasting.logger.logging import get_logger
from src.api.schemas import TrainingRequest, TrainingResponse
from fastapi import APIRouter
from pathlib import Path
from src.forecasting.pipelines.train_pipeline import ModelTrainer
from lightgbm import LGBMRegressor
from fastapi.exceptions import HTTPException

app = APIRouter()

data_path = Path("data/Walmart_final.csv")
save_model_path = Path("models/model_trained.pkl")

@app.post("/train", response_model=TrainingResponse)
async def train():
    try:
        trainer = ModelTrainer(data_path=data_path)
        best_model, metrics = trainer.run(save_model_path=save_model_path, model=LGBMRegressor)
        return TrainingResponse(
            status="201"

        )
    except:
        raise