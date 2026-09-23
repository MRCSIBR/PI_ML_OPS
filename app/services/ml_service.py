import os
import joblib
import logging
from typing import List, Dict, Any, Optional
from app.config import settings
from app.services.data_service import DataService

logger = logging.getLogger(__name__)

class MLRecommendationService:
    _instance: Optional["MLRecommendationService"] = None

    def __init__(self):
        self.artifacts: Optional[Dict[str, Any]] = None
        self.tfidf = None
        self.nn_model = None
        self.id_to_idx = {}
        self.is_model_loaded: bool = False

    @classmethod
    def get_instance(cls) -> "MLRecommendationService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_model(self):
        if self.is_model_loaded:
            return

        model_path = os.path.join(settings.DATA_PROCESSED_DIR, "recommendation_model.joblib")
        if os.path.exists(model_path):
            try:
                self.artifacts = joblib.load(model_path)
                self.tfidf = self.artifacts.get("tfidf")
                self.nn_model = self.artifacts.get("nn")
                self.id_to_idx = self.artifacts.get("id_to_idx", {})
                self.is_model_loaded = True
                logger.info("ML Recommendation model successfully loaded.")
            except Exception as e:
                logger.error(f"Error loading ML recommendation model: {e}")
        else:
            logger.warning(f"ML Model file not found at {model_path}")

    def recommend_by_item_id(self, item_id: str, top_n: int = 5) -> Optional[List[Dict[str, Any]]]:
        data_svc = DataService.get_instance()
        # 1. Fast path: precomputed cache
        cached = data_svc.get_game_recommendations(item_id)
        if cached:
            return cached[:top_n]

        # 2. Dynamic path if model is loaded and game exists in index
        if self.is_model_loaded and item_id in self.id_to_idx:
            idx = self.id_to_idx[item_id]
            games_df = data_svc.games_df
            if games_df is not None:
                feature_str = games_df.iloc[idx]["features"]
                vector = self.tfidf.transform([feature_str])
                dists, indices = self.nn_model.kneighbors(vector, n_neighbors=top_n + 1)
                
                recs = []
                for dist, n_idx in zip(dists[0][1:], indices[0][1:]):
                    g = games_df.iloc[n_idx]
                    recs.append({
                        "item_id": str(g["id"]),
                        "title": str(g["title"]),
                        "similarity": round(1.0 - float(dist), 4),
                        "genres": list(g["genres"]),
                        "price": float(g["price"])
                    })
                return recs

        return None

    def recommend_by_name(self, game_name: str, top_n: int = 5) -> Optional[Dict[str, Any]]:
        data_svc = DataService.get_instance()
        matched_id = data_svc.find_game_id_by_title(game_name)
        if not matched_id:
            return None
        
        recs = self.recommend_by_item_id(matched_id, top_n=top_n)
        return {
            "item_id": matched_id,
            "game_title": data_svc.get_game_title(matched_id),
            "recomendaciones": recs or []
        }

    def recommend_for_user(self, user_id: str, top_n: int = 5) -> Optional[List[Dict[str, Any]]]:
        data_svc = DataService.get_instance()
        if data_svc.reviews_df is None:
            return None

        # Find items the user recommended or rated positively
        user_reviews = data_svc.reviews_df[
            (data_svc.reviews_df["user_id"] == user_id) & 
            (data_svc.reviews_df["recommend"] == True)
        ]

        if user_reviews.empty:
            # Fallback to any reviews by this user
            user_reviews = data_svc.reviews_df[data_svc.reviews_df["user_id"] == user_id]

        if user_reviews.empty:
            return None

        user_item_ids = set(user_reviews["item_id"].astype(str).tolist())
        
        # Aggregate recommendations from user's liked games
        rec_candidates = {}
        for item_id in user_item_ids:
            recs = self.recommend_by_item_id(item_id, top_n=top_n)
            if recs:
                for r in recs:
                    rec_id = r["item_id"]
                    if rec_id not in user_item_ids:  # Don't recommend games already reviewed
                        if rec_id not in rec_candidates or r["similarity"] > rec_candidates[rec_id]["similarity"]:
                            rec_candidates[rec_id] = r

        sorted_recs = sorted(rec_candidates.values(), key=lambda x: x["similarity"], reverse=True)
        return sorted_recs[:top_n]
