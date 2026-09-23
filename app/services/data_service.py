import os
import json
import logging
import pandas as pd
from typing import Dict, Any, Optional, List
from app.config import settings

logger = logging.getLogger(__name__)

class DataService:
    _instance: Optional["DataService"] = None

    def __init__(self):
        self.games_df: Optional[pd.DataFrame] = None
        self.reviews_df: Optional[pd.DataFrame] = None
        self.developer_stats: Dict[str, Any] = {}
        self.developer_sentiment: Dict[str, Any] = {}
        self.genre_rankings: Dict[str, Any] = {}
        self.user_for_genre: Dict[str, Any] = {}
        self.user_stats: Dict[str, Any] = {}
        self.recommendations: Dict[str, Any] = {}
        self.game_id_to_title: Dict[str, str] = {}
        self.game_title_to_id: Dict[str, str] = {}
        self.is_loaded: bool = False

    @classmethod
    def get_instance(cls) -> "DataService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_data(self):
        if self.is_loaded:
            return

        logger.info("Initializing DataService and preloading datasets...")
        processed_dir = settings.DATA_PROCESSED_DIR

        # 1. Load games parquet
        games_parquet = os.path.join(processed_dir, "games.parquet")
        if os.path.exists(games_parquet):
            self.games_df = pd.read_parquet(games_parquet)
            self.game_id_to_title = self.games_df.set_index("id")["title"].to_dict()
            # Lowercase lookup for titles
            self.game_title_to_id = {
                row["title"].strip().lower(): row["id"]
                for _, row in self.games_df.iterrows()
            }
            logger.info(f"Loaded {len(self.games_df)} games.")

        # 2. Load reviews parquet
        reviews_parquet = os.path.join(processed_dir, "reviews.parquet")
        if os.path.exists(reviews_parquet):
            self.reviews_df = pd.read_parquet(reviews_parquet)
            logger.info(f"Loaded {len(self.reviews_df)} reviews.")

        # 3. Load precomputed JSON aggregations
        def load_json(filename: str):
            filepath = os.path.join(processed_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            return {}

        self.developer_stats = load_json("developer_stats.json")
        self.developer_sentiment = load_json("developer_sentiment.json")
        self.genre_rankings = load_json("genre_rankings.json")
        self.user_for_genre = load_json("user_for_genre.json")
        self.user_stats = load_json("user_stats.json")
        self.recommendations = load_json("recommendations.json")

        self.is_loaded = True
        logger.info("DataService initialized successfully.")

    # Helpers with case-insensitive matching
    def get_developer_stats(self, developer: str) -> Optional[List[Dict[str, Any]]]:
        if not developer:
            return None
        dev_clean = developer.strip()
        # Direct match
        if dev_clean in self.developer_stats:
            return self.developer_stats[dev_clean]
        # Case-insensitive match
        dev_lower = dev_clean.lower()
        for k, v in self.developer_stats.items():
            if k.lower() == dev_lower:
                return v
        return None

    def get_developer_sentiment(self, developer: str) -> Optional[Dict[str, int]]:
        if not developer:
            return None
        dev_clean = developer.strip()
        if dev_clean in self.developer_sentiment:
            return self.developer_sentiment[dev_clean]
        dev_lower = dev_clean.lower()
        for k, v in self.developer_sentiment.items():
            if k.lower() == dev_lower:
                return v
        return None

    def get_genre_rank(self, genre: str) -> Optional[Dict[str, Any]]:
        if not genre:
            return None
        g_clean = genre.strip()
        if g_clean in self.genre_rankings:
            return self.genre_rankings[g_clean]
        g_lower = g_clean.lower()
        for k, v in self.genre_rankings.items():
            if k.lower() == g_lower:
                return v
        return None

    def get_user_for_genre(self, genre: str) -> Optional[List[Dict[str, Any]]]:
        if not genre:
            return None
        g_clean = genre.strip()
        if g_clean in self.user_for_genre:
            return self.user_for_genre[g_clean]
        g_lower = g_clean.lower()
        for k, v in self.user_for_genre.items():
            if k.lower() == g_lower:
                return v
        return None

    def get_user_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        if not user_id:
            return None
        return self.user_stats.get(user_id.strip())

    def get_reviews_count(self, start_date: str, end_date: str) -> Dict[str, Any]:
        if self.reviews_df is None:
            return {"cantidad_usuarios": 0, "porcentaje_recomendacion": "0%"}
        
        # Filter by valid date
        mask = (self.reviews_df["date"] >= start_date) & (self.reviews_df["date"] <= end_date)
        filtered = self.reviews_df[mask]
        
        total_users = filtered["user_id"].nunique()
        if len(filtered) > 0:
            rec_pct = round(filtered["recommend"].mean() * 100, 2)
        else:
            rec_pct = 0.0

        return {
            "start_date": start_date,
            "end_date": end_date,
            "cantidad_usuarios": int(total_users),
            "porcentaje_recomendacion": f"{rec_pct}%"
        }

    def get_game_recommendations(self, item_id: str) -> Optional[List[Dict[str, Any]]]:
        return self.recommendations.get(str(item_id).strip())

    def get_game_title(self, item_id: str) -> str:
        return self.game_id_to_title.get(str(item_id).strip(), "Unknown Game")

    def find_game_id_by_title(self, title: str) -> Optional[str]:
        if not title:
            return None
        t_clean = title.strip().lower()
        if t_clean in self.game_title_to_id:
            return self.game_title_to_id[t_clean]
        # Partial match
        for k, v in self.game_title_to_id.items():
            if t_clean in k or k in t_clean:
                return v
        return None
