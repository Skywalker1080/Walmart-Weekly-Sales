from forecasting.logger.logging import get_logger
import numpy as np
import pandas as pd
from forecasting.stages.base import PipelineStage

logger = get_logger()

class FeatureMaker(PipelineStage):
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        pass

    def _model_date(self, df: pd.DataFrame):
        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['week'] = df['Date'].dt.isocalender().week.astype(int)
        df['Quarter'] = df['Date'].dt.quarter

        df = df.sort_values('Date')
        return df

    def _model_trend(self, df):
        pass
