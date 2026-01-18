from email.policy import HTTP
from fastapi import APIRouter, UploadFile, File
from fastapi.exceptions import HTTPException
import pandas as pd
from io import BytesIO
from src.forecasting.logger.logging import get_logger
from pathlib import Path
from src.api.schemas import PredictionResponse

router = APIRouter(prefix="/predictions", tags=["inference"])

logger = get_logger()

@router.post("")
async def predict(file: UploadFile = File(...), response_model=PredictionResponse):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    try:
        contents = await file.read()
        file_object = BytesIO(contents)

        from src.forecasting.pipelines.inference import InferencePipeline

        model_path = Path("models/model_trained.pkl")
        pipeline = InferencePipeline(model_path=model_path)
        predictions = pipeline.run(data_path=file_object, forecast_steps=12, save_plot=True)

        return {"status": "ok", "predictions": predictions.to_dict(orient="records")}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))