from forecasting.logger.logging import get_logger
import numpy as np
import pandas as pd
from forecasting.stages.base import PipelineStage

logger = get_logger()

class FeatureMaker(PipelineStage):
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Feature Maker: run started")
        df = df.copy()
        df = self._model_date(df)
        df = self._model_trend(df)
        df = self._calculate_rolling_mean(df)
        df = self._make_lag_features(df)
        df = self._validate_features(df)
        logger.info("Feature Maker: run complete")
        return df

    def _model_date(self, df: pd.DataFrame):
        logger.info("FeatureMaker: running _model_date")
        
        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['week'] = df['Date'].dt.isocalendar().week.astype(int)
        df['Quarter'] = df['Date'].dt.quarter

        df = df.sort_values('Date')
        logger.info("Feature Maker: run complete for _model_date") 
        return df

    def _model_trend(self, df: pd.DataFrame):
        logger.info("FeatureMaker: running _model_trend")
        df['Trend'] = np.arange(1, len(df)+1)
        logger.info("FeatureMaker: run complete for _model_trend")
        return df

    def _validate_features(self, df: pd.DataFrame):
        logger.info("FeatureMaker: running _validate_features")
        final_cols = ['Store', 'Date', 'Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price',
                      'CPI', 'Unemployment', 'year', 'month', 'week', 'Quarter', 'Trend', 'Rolling_mean_5',
                      'Rolling_mean_10', 'Lag_1', 'Lag_2']
        
        cols = df.columns

        if list(cols) != final_cols:
            logger.exception("Run failed for FeatureMaker: _validate_features")
            raise ValueError(f"Column order mismatch \n expected: {final_cols} \n Got instead: {cols}")
        
        logger.info("FeatureMaker: run complete for _validate_features")
        return df

    def _calculate_rolling_mean(self, df: pd.DataFrame):
        logger.info("FeatureMaker: running _calculate_rolling_mean")
        try:
            df['Rolling_mean_5'] = df['Weekly_Sales'].rolling(window=5).mean()
            df['Rolling_mean_10'] = df['Weekly_Sales'].rolling(window=10).mean()
            logger.info("FeatureMaker: run complete for _calculate_rolling_mean")
            return df
        except Exception:
            logger.exception("Failed to calculate rolling mean")
            raise

    def _make_lag_features(self, df:pd.DataFrame):
        logger.info("FeatureMaker: running _make_lag_features")
        try:
            df['Lag_1'] = df['Weekly_Sales'].shift(1)
            df['Lag_2'] = df['Weekly_Sales'].shift(2)
            df.bfill(inplace=True)
            logger.info("FeatureMaker: running _make_lag_features")
            return df
        except Exception:
            logger.exception("Failed to calculate lag feature")
            raise
        

    
        
        
