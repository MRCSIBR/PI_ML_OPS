from fastapi import APIRouter, HTTPException, Query
from app.services.data_service import DataService
from app.models.schemas import UserDataResponse, CountReviewsResponse

router = APIRouter()

@router.get(
    "/userdata/{User_id}",
    response_model=UserDataResponse,
    summary="Dinero gastado, porcentaje de recomendación y cantidad de items por usuario"
)
async def get_userdata(User_id: str):
    data_svc = DataService.get_instance()
    user_info = data_svc.get_user_data(User_id)
    if user_info is None:
        raise HTTPException(
            status_code=404,
            detail=f"Usuario '{User_id}' no encontrado en el sistema de reseñas/items."
        )
    return {
        "user_id": user_info["user_id"],
        "dinero_gastado": f"{user_info['money_spent_usd']} USD",
        "porcentaje_recomendacion": user_info["recommend_percentage"],
        "cantidad_de_items": user_info["items_count"]
    }

@router.get(
    "/countreviews",
    response_model=CountReviewsResponse,
    summary="Cantidad de usuarios que realizaron reviews y porcentaje de recomendación entre fechas dadas"
)
async def get_countreviews(
    start_date: str = Query(..., description="Fecha inicial en formato YYYY-MM-DD (ej: 2011-01-01)"),
    end_date: str = Query(..., description="Fecha final en formato YYYY-MM-DD (ej: 2015-12-31)")
):
    data_svc = DataService.get_instance()
    res = data_svc.get_reviews_count(start_date, end_date)
    return res
