from fastapi import APIRouter, HTTPException, Query
from typing import List
from app.services.data_service import DataService
from app.services.ml_service import MLRecommendationService
from app.models.schemas import GameRecommendationResponse, UserRecommendationResponse

router = APIRouter()

@router.get(
    "/recomendacion_juego/{item_id}",
    response_model=GameRecommendationResponse,
    summary="Recomienda 5 juegos similares basados en Machine Learning (Similitud del Coseno)"
)
async def recomendacion_juego(
    item_id: str,
    top_n: int = Query(5, ge=1, le=20, description="Número de recomendaciones a retornar")
):
    data_svc = DataService.get_instance()
    ml_svc = MLRecommendationService.get_instance()

    recs = ml_svc.recommend_by_item_id(item_id, top_n=top_n)
    if not recs:
        raise HTTPException(
            status_code=404,
            detail=f"Juego con item_id '{item_id}' no encontrado en el modelo de recomendación."
        )

    return {
        "item_id": item_id,
        "game_title": data_svc.get_game_title(item_id),
        "recomendaciones": recs
    }

@router.get(
    "/recomendacion_juego_por_nombre",
    response_model=GameRecommendationResponse,
    summary="Búsqueda y recomendación de 5 juegos similares por título"
)
async def recomendacion_juego_por_nombre(
    title: str = Query(..., description="Nombre del videojuego a buscar (ej: Counter-Strike o Portal)"),
    top_n: int = Query(5, ge=1, le=20)
):
    ml_svc = MLRecommendationService.get_instance()
    res = ml_svc.recommend_by_name(title, top_n=top_n)
    if not res:
        raise HTTPException(
            status_code=404,
            detail=f"No se encontró ningún juego similar al título '{title}'."
        )
    return res

@router.get(
    "/recomendacion_usuario/{user_id}",
    response_model=UserRecommendationResponse,
    summary="Recomienda 5 juegos para un usuario según sus preferencias históricas"
)
async def recomendacion_usuario(
    user_id: str,
    top_n: int = Query(5, ge=1, le=20)
):
    ml_svc = MLRecommendationService.get_instance()
    recs = ml_svc.recommend_for_user(user_id, top_n=top_n)
    if not recs:
        raise HTTPException(
            status_code=404,
            detail=f"Usuario '{user_id}' no posee suficiente historial de reseñas para generar recomendaciones personalizadas."
        )
    return {
        "user_id": user_id,
        "recomendaciones": recs
    }
