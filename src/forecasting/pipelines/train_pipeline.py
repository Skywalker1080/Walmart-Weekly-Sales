from forecasting.logger.logging import get_logger
from forecasting.utils.eval import evaluate_model
from pathlib import Path
import pandas as pd
from lightgbm import LGBMRegressor
import pickle
from typing import Dict, Optional, Union, Type
import mlflow
import mlflow.lightgbm

from sklearn.model_selection import train_test_split

logger = get_logger()

Best_params = {'n_estimators': 1077, 'num_leaves': 82, 'max_bin': 118}


data_path = Path("data/Walmart_final.csv")

class ModelTrainer:
    def __init__(self, data_path):
        self.data_path = data_path

    def run(self,
            save_model_path,
            model: Optional[Union[LGBMRegressor, Type[LGBMRegressor]]] = None,) -> Dict:
        """Load data, split, train and optionally save model. Returns dict with model and metrics."""
        logger.info("Starting Model Trainer")

        df = self._get_data(self.data_path)
        X_train, X_test, y_train, y_test = self._split_dataset(df)

        # If caller passed the model class (e.g., LGBMRegressor) instantiate it so
        # we call instance methods like fit/predict correctly.
        if model is None:
            model = LGBMRegressor(**Best_params,random_state=42, n_jobs=-1)
        elif isinstance(model, type):
            model = model(random_state=42)

        mlflow.set_experiment("walmart_weekly_sales_forecasting")
        with mlflow.start_run(run_name="lightbgm_train"):
            mlflow.log_param("Best_params", Best_params)
            mlflow.log_param("test_size", 0.25)

            trained_model, metrics = self._train(X_train, X_test, y_train, y_test,
                                                model, save_model_path, save_model=False)

            mlflow.log_metrics({
                "train_mae": metrics["train"]["MAE"],
                "train_rmse": metrics["train"]["RMSE"],
                "train_r2": metrics["train"]["R2"],
                "test_mae": metrics["test"]["MAE"],
                "test_rmse": metrics["test"]["RMSE"],
                "test_r2": metrics["test"]["R2"],
            })

            mlflow.lightgbm.log_model(trained_model)

        logger.info("Model training finished")
        return {"metrics": metrics}

    def _get_data(self, path: Path):
        
        try:
            logger.info(f"Loading CSV file from {path}")
            data = pd.read_csv(path)
            logger.info(f"Succesfully loaded CSV from {path}")
            return data
        except Exception:
            logger.exception(f"Failed to load data from {path}")
            raise

    def _split_dataset(self, df: pd.DataFrame, test_size: float = 0.25):
        
        df = df.copy()
        X = df.drop(columns=['Date', 'Weekly_Sales'])
        y = df['Weekly_Sales']

        if y.isnull().any():
            logger.exception("Missin values in target variable")
            raise ValueError("Missing values in target variable 'Weekly_Sales'")
        logger.info("Splitting dataset")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False, random_state=42)
        logger.info("Splitting Complete")
        logger.debug(f"X_train: {len(X_train)}, X_test: {len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    def _train(self, X_train, X_test, y_train, y_test, model: LGBMRegressor, model_save_path, save_model = True):
        logger.info("Training Model")
        model = model
        # Fit and predict expect the feature arrays as the first positional argument
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_mae, train_rmse, train_r2 = evaluate_model(y_train, y_train_pred)
        test_mae, test_rmse, test_r2 = evaluate_model(y_test, y_test_pred)

        metrics = {
            "train": {"MAE": train_mae, "RMSE": train_rmse, "R2": train_r2},
            "test": {"MAE": test_mae, "RMSE": test_rmse, "R2": test_r2},
        }
        
        logger.debug(f"metrics: {metrics['train'], metrics['test']}")

        if save_model:
            try:
                model_save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(model_save_path, "wb") as f:
                    pickle.dump(model, f)
                logger.info(f"Saved model to {model_save_path}")
            except Exception:
                logger.exception("save_model failed")

        return model, metrics


        
        

        