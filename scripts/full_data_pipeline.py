from forecasting.pipelines.data_pipeline import run_pipeline
from pathlib import Path
from forecasting.utils.save_data import save_df_to_csv
from forecasting.logger.logging import get_logger

logger = get_logger()

path_Str = Path("data/Walmart.csv")
output_path = Path("src/forecasting/data/Walmart_final.csv")

df = run_pipeline(path=path_Str)
print(df)
logger.info("Saving data from df to csv")
save_df_to_csv(output_path=output_path, df=df)