from fastapi import APIRouter
from app.api.v1.endpoints import games, users, reviews, recommendations

api_router = APIRouter()

api_router.include_router(games.router, tags=["Games & Developers"])
api_router.include_router(users.router, tags=["Users & Reviews"])
api_router.include_router(reviews.router, tags=["Sentiment Analysis"])
api_router.include_router(recommendations.router, tags=["Machine Learning Recommendations"])
