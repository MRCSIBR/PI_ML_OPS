from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any
from app.services.data_service import DataService
from app.models.schemas import DeveloperYearStat, GenreRankResponse, UserForGenreResponse

router = APIRouter()

@router.get(
    "/developer/{desarrollador}",
    response_model=List[DeveloperYearStat],
    summary="Estadísticas de contenido Free e items por año según empresa desarrolladora"
)
async def get_developer_stats(desarrollador: str):
    data_svc = DataService.get_instance()
    stats = data_svc.get_developer_stats(desarrollador)
    if stats is None or len(stats) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Desarrollador '{desarrollador}' no encontrado en el catálogo."
        )
    return stats

@router.get(
    "/genre/{genre}",
    response_model=GenreRankResponse,
    summary="Puesto en el ranking de un género"
)
async def get_genre_rank(genre: str):
    data_svc = DataService.get_instance()
    rank_info = data_svc.get_genre_rank(genre)
    if rank_info is None:
        raise HTTPException(
            status_code=404,
            detail=f"Género '{genre}' no encontrado en el catálogo."
        )
    return {
        "genre": genre,
        "puesto_ranking": rank_info["rank"],
        "total_juegos": rank_info["count"]
    }

@router.get(
    "/userforgenre/{genre}",
    response_model=UserForGenreResponse,
    summary="Top 5 usuarios con mayor interacción para un género dado"
)
async def get_user_for_genre(genre: str):
    data_svc = DataService.get_instance()
    users = data_svc.get_user_for_genre(genre)
    if users is None:
        raise HTTPException(
            status_code=404,
            detail=f"No se encontraron interacciones para el género '{genre}'."
        )
    return {
        "genre": genre,
        "top_5_users": users
    }
