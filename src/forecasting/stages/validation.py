from forecasting.logger.logging import get_logger
import numpy as np
import pandas as pd
from forecasting.stages.base import PipelineStage

logger = get_logger()

class Validation(PipelineStage):
    def run(self, df):
        self._check_time(df)
        self._check_target(df)
        return df

    def _check_time(self, df: pd.DataFrame) -> None:
        if not isinstance(df['Date'], np.datetime64):
            logger.debug("Date should be in datetime format")
            raise ValueError("Date column must be in np.datetime64")
        elif not df['Date'].is_monotonic_increasing:
            raise ValueError("Date column is not strictly increasing")
        
    def _check_target(self, df: pd.DataFrame) -> None:
        logger.debug("Valiadtion: checking target variable")
        y = df["Weekly_Sales"]

        if y.isna().any():
            logger.debug("Target column has missing values")
            raise ValueError("Target column has missing values")
        
        if y.nunique() <= 1:
            logger.debug("Target is constant or near-constant")
            raise ValueError("Target is constant or near-constant")
    
