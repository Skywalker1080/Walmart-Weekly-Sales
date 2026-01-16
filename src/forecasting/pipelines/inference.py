# inference pipeline
import pandas as pd
from forecasting.logger.logging import get_logger
from forecasting.stages.ingestion import Ingestion
from typing import Union
from pathlib import Path
import pickle
from forecasting.stages.validation import Validation
from forecasting.stages.modelling import FeatureMaker
import matplotlib.pyplot as plt
from datetime import datetime


logger = get_logger()

class InferencePipeline:
    def __init__(self, model_path: Union[str, Path]):
        """Execute full inference pipeline: ingest -> validate -> engineer features -> predict"""
        
        self.model_path = Path(model_path)
        self.model = self._load_model()
        self.ingestion = Ingestion()
        self.validation = Validation()
        self.feature_maker = FeatureMaker()
    
    def run(self, data_path: Union[str, Path], forecast_steps: int = 12, save_plot: bool = True, plot_dir: Union[str, Path] = "results") -> pd.DataFrame:
        """Execute full inference pipeline with multi-step forecasting"""
        logger.info(f"Inference: Starting pipeline for {forecast_steps} week forecast")

        df = self.ingestion.run(path=str(data_path))
        df = self.validation.run(df)
        df = self.feature_maker.run(df)

        predictions = self._predict(df, forecast_steps=forecast_steps)
        
        if save_plot:
            self._plot_predictions(predictions, plot_dir)
        
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

    def _predict(self, df: pd.DataFrame, forecast_steps: int = 12) -> pd.DataFrame:
        """Generate multi-step predictions (12 weeks ahead)"""
        try:
            logger.info(f"Inference: Multi-step forecasting for {forecast_steps} weeks ahead")
            
            all_predictions = []
            current_df = df.copy()
            
            for step in range(forecast_steps):
                logger.debug(f"Inference: Forecasting step {step + 1}/{forecast_steps}")
                
                # Drop Weekly_Sales for prediction 
                X = current_df.drop(columns=['Date', 'Weekly_Sales'])
                
                # Make prediction for this step
                step_prediction = self.model.predict(X)
                
                
                result_step = current_df[['Date', 'Store']].copy()
                result_step['predicted_sales'] = step_prediction
                result_step['forecast_step'] = step + 1
                all_predictions.append(result_step)
                
                
                # Shift date forward by 1 week
                current_df['Date'] = pd.to_datetime(current_df['Date']) + pd.Timedelta(weeks=1)
                
                
                current_df['Weekly_Sales'] = step_prediction
                
                # Re-engineer features for next step
                if step < forecast_steps - 1:  # Dont reengineer on last step
                    current_df = self.feature_maker.run(current_df)
            
            # Combine all predictions
            result = pd.concat(all_predictions, ignore_index=True)
            
            logger.info(f"Inference: Multi-step predictions generated successfully for {forecast_steps} weeks")
            return result
        except Exception:
            logger.exception("Multi-step prediction failed")
            raise

    def _plot_predictions(self, predictions: pd.DataFrame, plot_dir: Union[str, Path] = "results") -> None:
        """Plot and save prediction charts"""
        try:
            plot_dir = Path(plot_dir)
            plot_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info("Inference: Plotting predictions")
            
            
            stores = predictions['Store'].unique()
            
            
            for store in stores:
                store_data = predictions[predictions['Store'] == store].copy()
                store_data['Date'] = pd.to_datetime(store_data['Date'])
                store_data = store_data.sort_values('Date')
                
                plt.figure(figsize=(12, 6))
                plt.plot(store_data['Date'], store_data['predicted_sales'], 
                        marker='o', linewidth=2, markersize=8, label='Predicted Sales', color='#1f77b4')
                
                plt.xlabel('Date', fontsize=12, fontweight='bold')
                plt.ylabel('Weekly Sales', fontsize=12, fontweight='bold')
                plt.title(f'Store {store} - 12 Week Sales Forecast', fontsize=14, fontweight='bold')
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, alpha=0.3, linestyle='--')
                plt.legend(fontsize=11)
                plt.tight_layout()
                
               
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_path = plot_dir / f"forecast_store_{store}_{timestamp}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"Inference: Plot saved for Store {store} at {plot_path}")
                plt.close()
            
            logger.info(f"Inference: All prediction plots saved to {plot_dir}")
        except Exception:
            logger.exception("Failed to plot predictions")
            raise