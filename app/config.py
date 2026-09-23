import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "Steam Games ML Recommendation & Analytics API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    DESCRIPTION: str = (
        "Production-ready MLOps API providing game recommendations using Machine Learning "
        "(TF-IDF + Cosine Similarity) and exploratory analytics on Steam catalog & reviews."
    )
    
    BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DATA_PROCESSED_DIR: str = os.path.join(BASE_DIR, "data", "processed")
    
    model_config = SettingsConfigDict(case_sensitive=True)

settings = Settings()
