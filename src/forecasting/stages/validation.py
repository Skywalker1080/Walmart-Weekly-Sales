from forecasting.logger.logging import get_logger
import numpy as np
import pandas as pd
from forecasting.stages.base import PipelineStage

logger = get_logger()

class Validation(PipelineStage):
    def run(self, df):
        self._check_time(df)
        self._check_target(df)
        self._check_features(df)
        return df

    def _check_time(self, df: pd.DataFrame) -> None:
        if 'Date' not in df.columns:
            raise KeyError("Required column 'Date' is missing ")
        
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            logger.debug("Date should be in datetime format")
            raise TypeError("Date column must be in np.datetime64")
        elif not df['Date'].is_monotonic_increasing:
            raise ValueError("Date column is not strictly increasing")
        
    def _check_target(self, df: pd.DataFrame) -> None:
        logger.debug("Valiadtion: checking target variable")
        y = df["Weekly_Sales"]

        if y.empty:
            logger.exception("Missing required column 'Weekly_Sales'")
            raise KeyError("Missing required column 'Weekly_Sales'")

        if y.isna().any():
            logger.debug("Target column has missing values")
            raise ValueError("Target column has missing values")
        
        if y.nunique() <= 1:
            logger.debug("Target is constant or near-constant")
            raise ValueError("Target is constant or near-constant")
        
    def _check_features(self, df: pd.DataFrame):

        if not pd.api.types.is_numeric_dtype(df["Weekly_Sales"]):
            raise TypeError("'Weekly_Sales' must be numeric")
    
