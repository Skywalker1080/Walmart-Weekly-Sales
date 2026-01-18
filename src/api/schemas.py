from pydantic import BaseModel
from typing import List, Dict
from datetime import date

class Health(BaseModel):
    """Schema for api health check"""
    status: str

class TrainingRequest(BaseModel):
    """Schema for training request"""
    data_path: str
    model_save_path: str

class MetricValues(BaseModel):
    MAE: float
    RMSE: float
    R2: float

class TrainingMetrics(BaseModel):
    train: MetricValues
    test: MetricValues

class TrainingResponse(BaseModel):
    """Schema for training response"""
    status: str
    message: str
    metrics: TrainingMetrics

class PredictionRequest(BaseModel):
    """Schema for prediction request"""
    data_path: str
    forecast_steps: int = 12
    save_plot: bool = True

class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    status: str
    message: str
    plot_location: str