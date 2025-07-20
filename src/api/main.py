from fastapi import FastAPI
from ..ml_pipeline.serving.api import app as ml_api

app = FastAPI()

@app.get("/health")
async def health_check():
    return {"status": "ok"}

app.include_router(ml_api.router, prefix="/ml", tags=["ml"])