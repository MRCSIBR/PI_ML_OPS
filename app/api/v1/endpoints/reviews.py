from fastapi import APIRouter, HTTPException
from app.services.data_service import DataService
from app.models.schemas import SentimentAnalysisResponse

router = APIRouter()

@router.get(
    "/sentiment_analysis/{empresa_desarrolladora}",
    response_model=SentimentAnalysisResponse,
    summary="Cantidad de reseñas categorizadas según análisis de sentimiento para una empresa desarrolladora"
)
async def get_sentiment_analysis(empresa_desarrolladora: str):
    data_svc = DataService.get_instance()
    sentiment = data_svc.get_developer_sentiment(empresa_desarrolladora)
    if sentiment is None:
        raise HTTPException(
            status_code=404,
            detail=f"Empresa desarrolladora '{empresa_desarrolladora}' no encontrada con reseñas analizadas."
        )
    return {
        "desarrollador": empresa_desarrolladora,
        "Negative": sentiment.get("Negative", 0),
        "Neutral": sentiment.get("Neutral", 0),
        "Positive": sentiment.get("Positive", 0)
    }
