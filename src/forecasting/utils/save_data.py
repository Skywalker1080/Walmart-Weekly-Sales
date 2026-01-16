from pathlib import Path
import pandas as pd
from forecasting.logger.logging import get_logger

logger = get_logger()

def save_df_to_csv(output_path: Path, df: pd.DataFrame, index: bool = False, overwrite: bool = True):
    
    if not isinstance(output_path, Path):
        logger.exception("output_path must be of pathlib.Path")
        raise TypeError("output_path must be of pathlib.Path")
    
    if output_path.suffix != '.csv':
        logger.exception("Path suffix must be .csv, Got: {output_path.suffix}")
        raise ValueError(f"Path suffix must be .csv, Got: {output_path.suffix}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"File already exists: {output_path}"
        )
    
    try:
        df.to_csv(output_path, index=index)
        logger.info(f"Data saved successfully to {output_path}")
    except Exception:
        logger.exception("Failed to save DataFrame to %s", output_path)
        raise

    return output_path

