from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.model_service import ModelService
from app.routers import health, predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load both saved local-model bundles once; request handlers never retrain or reload them.
    service = ModelService(settings.model_run_dir, settings.processed_dir)
    service.load()
    app.state.model_service = service
    yield


app = FastAPI(
    title="Pinnacle Diabetes Prediction API",
    description="Separate local EHR and Pima XGBoost prediction services. Research use only; not clinically validated.",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(predict.router, prefix="/api")
