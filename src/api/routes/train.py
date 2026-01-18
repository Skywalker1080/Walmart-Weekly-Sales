from src.forecasting.logger.logging import get_logger
from src.api.schemas import TrainingRequest, TrainingResponse, TrainingRequest
from fastapi import APIRouter
from pathlib import Path
from src.forecasting.pipelines.train_pipeline import ModelTrainer
from lightgbm import LGBMRegressor
from fastapi.exceptions import HTTPException

app = APIRouter(prefix="/models", tags=["training"])

data_path = Path("data/Walmart_final.csv")
save_model_path = Path("models/model_trained.pkl")

@app.post("/train", response_model=TrainingResponse)
async def train(payload: TrainingRequest):
    try:
        trainer = ModelTrainer(data_path=data_path)
        result = trainer.run(save_model_path=save_model_path, model=LGBMRegressor)
        return TrainingResponse(
            status="201",
            message="Training completed successfully",
            metrics= result['metrics'],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))