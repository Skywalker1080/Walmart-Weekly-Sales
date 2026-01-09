from forecasting.stages.ingestion import Ingestion
from forecasting.stages.validation import Validation
from forecasting.stages.modelling import FeatureMaker

def run_pipeline(path: str):
    df = Ingestion().run(path=path)
    df = Validation().run(df)
    df = FeatureMaker().run(df)
    return df