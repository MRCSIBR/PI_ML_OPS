import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.services.data_service import DataService
from app.services.ml_service import MLRecommendationService
from app.api.v1.api import api_router
from app.models.schemas import HealthResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s"
)
logger = logging.getLogger("SteamMLOps")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler:
    Preloads all cleaned datasets, lookup tables, and the ML recommendation model
    into memory on server startup to guarantee sub-millisecond API response times.
    """
    logger.info("Application starting up... Loading datasets and ML models.")
    data_svc = DataService.get_instance()
    data_svc.load_data()

    ml_svc = MLRecommendationService.get_instance()
    ml_svc.load_model()
    
    logger.info("Application ready to receive requests.")
    yield
    logger.info("Application shutting down...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoints
@app.get("/", tags=["General"])
async def root():
    return {
        "message": "Bienvenido a la API de Recomendación y Análisis de Videojuegos de Steam (MLOps MVP)",
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/health",
        "endpoints_disponibles": {
            "developer": "/developer/{desarrollador}",
            "userdata": "/userdata/{User_id}",
            "countreviews": "/countreviews?start_date=YYYY-MM-DD&end_date=YYYY-MM-DD",
            "genre": "/genre/{genre}",
            "userforgenre": "/userforgenre/{genre}",
            "sentiment_analysis": "/sentiment_analysis/{empresa_desarrolladora}",
            "recomendacion_juego (ML Item-Item)": "/recomendacion_juego/{item_id}",
            "recomendacion_juego_por_nombre (ML)": "/recomendacion_juego_por_nombre?title=GameTitle",
            "recomendacion_usuario (ML User-Item)": "/recomendacion_usuario/{user_id}"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    data_svc = DataService.get_instance()
    ml_svc = MLRecommendationService.get_instance()
    
    catalog_count = len(data_svc.games_df) if data_svc.games_df is not None else 0
    users_count = len(data_svc.user_stats)
    
    return {
        "status": "healthy" if data_svc.is_loaded else "initializing",
        "games_catalog_count": catalog_count,
        "users_count": users_count,
        "recommendation_model_loaded": ml_svc.is_model_loaded
    }

# Mount routers directly at root for project specifications compatibility
app.include_router(api_router)

# Mount also under /api/v1 for standard API versioning
app.include_router(api_router, prefix=settings.API_V1_STR)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
