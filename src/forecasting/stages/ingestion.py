import numpy as np
import pandas as pd
from forecasting.logger.logging import get_logger
from forecasting.stages.base import PipelineStage


logger = get_logger()
    
class Ingestion(PipelineStage):
    def run(self, path: str = None, data: pd.DataFrame = None, file_object = None):
        """Read CSV, parse Date, sort by Date. Logs progress and errors.""" # assumes source data is in CSV format

        logger.info("Ingest: loading data")

        try:
            if isinstance(data, pd.DataFrame):
                df = data.copy()
                logger.debug("Ingestion: Data received as DataFrame")
            elif file_object is not None:

                df = pd.read_csv(file_object)
                logger.debug("Ingestion: Data loaded from file upload")
            
            else:
                df = pd.read_csv(path)
                logger.debug(f"Ingestion: Data Loaded from {path}")

            # Parse Date robustly: try day-first formats (e.g. '19-02-2010'), then fall back to inferred formats
            try:
                df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, infer_datetime_format=True, errors='raise')
                logger.debug("Ingest: Converted 'Date' to datetime dtype (dayfirst=True)")
            except Exception:
                # Fallback: try without dayfirst to support month-first and ISO formats
                df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True, errors='raise')
                logger.debug("Ingest: Converted 'Date' to datetime dtype (dayfirst=False fallback)")
        
            df['Date'] = pd.to_datetime(df['Date'], errors='raise')
            logger.debug("Ingest: Converted 'Date' to datetime dtype")

            df = df.sort_values("Date").reset_index(drop=True)
            logger.debug("Ingest: completed successfully; final shape=%s", df.shape)

            return df
        except Exception:
            logger.exception("Ingest: failed to ingest data or parse data from %s", path)
            raise
    

    