from forecasting.data.ingestion import ingest_data

path = "data/walmart_ml_data.csv"

df = ingest_data(path)
print(df.head())