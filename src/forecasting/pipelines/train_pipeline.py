from forecasting.logger.logging import get_logger
from forecasting.utils.save_model import save_model
from forecasting.utils.save_data import get_data
from pathlib import Path
import pandas as pd
from lightgbm import LGBMRegressor

from sklearn.model_selection import train_test_split

logger = get_logger()

class ModelTrainer:
    def run(self):
        pass

    def _get_data(self, path: Path):
        
        try:
            data = get_data(path=path)
            return data
        except Exception:
            logger.exception(f"Failed to load data from {path}")
            raise

    def _split_dataset(self, df: pd.DataFrame, test_size: float = 0.25):
        df = df.copy()
        X = df.drop(columns=['Date', 'Weekly_Sales'])
        y = df['Weekly_Sales']

        if y.empty:
            raise ValueError("Missing values in target variable 'Weekly_Sales'")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False, random_state=42)
        return X_train, X_test, y_train, y_test
    
    def _train(X_train, X_test, y_train, y_test, model: LGBMRegressor):
        model = model
        model.fit()
        pass
        # implementation pending
        