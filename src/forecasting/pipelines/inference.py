# inference pipeline
import pandas as pd
from forecasting.logger.logging import get_logger
from forecasting.stages.ingestion import Ingestion
from typing import Union
from pathlib import Path
import pickle
from forecasting.stages.validation import Validation
from forecasting.stages.modelling import FeatureMaker


logger = get_logger()

class InferencePipeline:
    def __init__(self, model_path: Union[str, Path]):
        """Execute full inference pipeline: ingest -> validate -> engineer features -> predict"""
        
        self.model_path = Path(model_path)
        self.model = self._load_model()
        self.ingestion = Ingestion()
        self.validation = Validation()
        self.feature_maker = FeatureMaker()
    
    def run(self, data_path: Union[str, Path]) -> pd.DataFrame:
        """Execute full inference pipeline"""
        logger.info("Inference: Starting pipeline")

        df = self.ingestion.run(path=str(data_path))
        #df = self.validation.run(df)
        df = self.feature_maker.run(df)

        predictions = self._predict(df)
        logger.info("Inference: Pipeline Complete")
        return predictions

    def _load_model(self):
        """Inference: Loading trained model from disk"""
        try:
            logger.debug(F"Loading model from {self.model_path}")
            with open(self.model_path, "rb") as f:
                model = pickle.load(f)
            logger.info(f"Model loaded sucessfully from {self.model_path}")
            return model
        except Exception:
            logger.exception(f"Failed to load model from {self.model_path}")
            raise

    def _predict(self, df:pd.DataFrame) -> pd.DataFrame:
        """Generate predictions on data"""
        try:
            logger.log(f"Inference: predicting for {len(df)} records")

            X = df.drop(columns=['Date', 'Weekly_Sales'])

            predictions = self.model.predict(X)

            result = df[['Date', 'Store']].copy()
            result['predicted_sales'] = predictions
            result['actual_sales'] = df['Weekly_Sales']

            logger.info(f"Inference: Predictions genereted succesfully")
            return result
        except Exception:
            logger.exception("Prediction failed")
            raise