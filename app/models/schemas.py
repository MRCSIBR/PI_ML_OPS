from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

class DeveloperYearStat(BaseModel):
    Año: int = Field(..., description="Año de lanzamiento")
    Cantidad_de_Items: int = Field(..., alias="Cantidad de Items", description="Total de juegos lanzados ese año")
    Contenido_Free: str = Field(..., alias="Contenido Free", description="Porcentaje de juegos gratuitos")

    model_config = ConfigDict(populate_by_name=True)

class UserDataResponse(BaseModel):
    user_id: str
    dinero_gastado: str
    porcentaje_recomendacion: str
    cantidad_de_items: int

class CountReviewsResponse(BaseModel):
    start_date: str
    end_date: str
    cantidad_usuarios: int
    porcentaje_recomendacion: str

class GenreRankResponse(BaseModel):
    genre: str
    puesto_ranking: int
    total_juegos: int

class TopUserForGenre(BaseModel):
    user_id: str
    user_url: str
    interactions_count: int

class UserForGenreResponse(BaseModel):
    genre: str
    top_5_users: List[TopUserForGenre]

class SentimentAnalysisResponse(BaseModel):
    desarrollador: str
    Negative: int
    Neutral: int
    Positive: int

class GameRecommendationItem(BaseModel):
    item_id: str
    title: str
    similarity: float
    genres: Optional[List[str]] = None
    price: Optional[float] = None

class GameRecommendationResponse(BaseModel):
    item_id: str
    game_title: str
    recomendaciones: List[GameRecommendationItem]

class UserRecommendationResponse(BaseModel):
    user_id: str
    recomendaciones: List[GameRecommendationItem]

class HealthResponse(BaseModel):
    status: str
    games_catalog_count: int
    users_count: int
    recommendation_model_loaded: bool
