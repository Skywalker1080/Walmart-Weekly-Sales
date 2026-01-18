from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from src.forecasting.logger.logging import get_logger
from src.api.routes.train import app as train_router
from src.api.routes.predict import router

logger = get_logger()
app = FastAPI(title="Walmart Sales Forecast API")
app.include_router(train_router)
app.include_router(router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__=="__main__":
    uvicorn.run(app, port=8000, reload=True)