"""
Data Processing, Feature Engineering & ML Recommendation Pipeline for Steam MLOps.

This script:
1. Cleans and structures steam_games.parquet
2. Unnests user_reviews.parquet and performs NLP Sentiment Analysis (0=negative, 1=neutral, 2=positive)
3. Cleans review dates and joins game metadata
4. Trains an item-item ML recommendation system (TF-IDF + NearestNeighbors with Cosine Similarity)
5. Pre-computes and serializes optimized artifacts in data/processed/ for sub-millisecond production inference.
"""

import os
import re
import json
import logging
import joblib
import numpy as np
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def clean_games(games_path: str) -> pd.DataFrame:
    logging.info(f"Loading games dataset from {games_path}...")
    df = pd.read_parquet(games_path)

    # Filter null or invalid IDs
    df = df.dropna(subset=["id"]).copy()
    df["id"] = df["id"].astype(str).str.strip()
    df = df[df["id"] != ""]
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)

    # Clean app_name / title
    df["title"] = df["title"].fillna(df["app_name"]).fillna("Unknown Game").astype(str)
    df["app_name"] = df["app_name"].fillna(df["title"]).astype(str)

    # Clean price
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)

    # Clean release date and extract year
    df["release_date"] = df["release_date"].astype(str)
    df["year"] = df["release_date"].str.extract(r"(\b\d{4}\b)")[0]
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # Developer and publisher
    df["developer"] = df["developer"].fillna("Unknown Developer").astype(str).str.strip()
    df["publisher"] = df["publisher"].fillna("Unknown Publisher").astype(str).str.strip()

    # Normalize genres and tags
    def normalize_list(val):
        if isinstance(val, (list, np.ndarray)):
            return [str(x).strip() for x in val if str(x).strip()]
        if isinstance(val, str) and val.startswith("["):
            try:
                import ast
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                pass
        return []

    df["genres"] = df["genres"].apply(normalize_list)
    df["tags"] = df["tags"].apply(normalize_list)

    # Feature string for ML recommendation
    def build_feature_str(row):
        g = " ".join(row["genres"])
        t = " ".join(row["tags"])
        d = row["developer"].replace(" ", "_")
        return f"{g} {t} {d}".lower()

    df["features"] = df.apply(build_feature_str, axis=1)

    logging.info(f"Cleaned games shape: {df.shape}")
    return df


def clean_reviews_and_sentiment(reviews_path: str) -> pd.DataFrame:
    logging.info(f"Loading user reviews dataset from {reviews_path}...")
    ur = pd.read_parquet(reviews_path)

    # Explode the list of review dictionaries
    exploded = ur.explode("reviews").dropna(subset=["reviews"]).copy()
    reviews_df = pd.json_normalize(exploded["reviews"])
    reviews_df["user_id"] = exploded["user_id"].values
    reviews_df["user_url"] = exploded["user_url"].values

    # Clean item_id
    reviews_df["item_id"] = reviews_df["item_id"].astype(str).str.strip()
    reviews_df = reviews_df[reviews_df["item_id"] != ""].copy()

    # Clean recommend flag
    reviews_df["recommend"] = reviews_df["recommend"].astype(bool)

    # Date parsing: e.g. "Posted November 5, 2011." or "Posted February 3."
    def parse_posted_date(val):
        if not isinstance(val, str):
            return None
        m = re.search(r"Posted\s+([A-Za-z]+\s+\d{1,2}(?:,\s+\d{4})?)", val)
        if not m:
            return None
        date_str = m.group(1).replace(".", "").strip()
        # If year missing, append 2016 (year the dataset was collected)
        if not re.search(r"\d{4}", date_str):
            date_str = f"{date_str}, 2016"
        try:
            return pd.to_datetime(date_str).strftime("%Y-%m-%d")
        except Exception:
            return None

    reviews_df["date"] = reviews_df["posted"].apply(parse_posted_date)

    # NLP Sentiment Analysis
    # 0 = bad/negative, 1 = neutral or empty, 2 = positive
    logging.info("Computing sentiment analysis on user reviews...")
    def analyze_sentiment(text):
        if not isinstance(text, str) or not text.strip():
            return 1
        try:
            polarity = TextBlob(text).sentiment.polarity
            if polarity > 0.1:
                return 2
            elif polarity < -0.1:
                return 0
            else:
                return 1
        except Exception:
            return 1

    reviews_df["sentiment_analysis"] = reviews_df["review"].apply(analyze_sentiment)
    logging.info(f"Cleaned reviews shape: {reviews_df.shape}")
    logging.info(f"Sentiment distribution:\n{reviews_df['sentiment_analysis'].value_counts()}")
    return reviews_df


def train_and_export_recommender(games_df: pd.DataFrame):
    logging.info("Training Content-Based ML Recommender (TF-IDF + Cosine Similarity)...")
    tfidf = TfidfVectorizer(stop_words="english", max_features=3000)
    X = tfidf.fit_transform(games_df["features"])

    nn = NearestNeighbors(n_neighbors=6, metric="cosine", algorithm="brute")
    nn.fit(X)

    # Precompute top-5 recommendations for all games for O(1) latency
    logging.info("Precomputing top 5 recommendations for all catalog games...")
    dists, indices = nn.kneighbors(X)

    recommendations = {}
    id_to_idx = {row["id"]: i for i, row in games_df.iterrows()}

    for i, row in games_df.iterrows():
        item_id = row["id"]
        rec_list = []
        for dist, idx in zip(dists[i][1:], indices[i][1:]):
            rec_game = games_df.iloc[idx]
            sim = 1.0 - float(dist)
            rec_list.append({
                "item_id": rec_game["id"],
                "title": rec_game["title"],
                "similarity": round(sim, 4),
                "genres": rec_game["genres"],
                "price": float(rec_game["price"])
            })
        recommendations[item_id] = rec_list

    # Save precomputed recommendations
    rec_path = os.path.join(PROCESSED_DIR, "recommendations.json")
    with open(rec_path, "w", encoding="utf-8") as f:
        json.dump(recommendations, f, ensure_ascii=False)
    logging.info(f"Saved precomputed recommendations to {rec_path}")

    # Save model artifacts for dynamic queries
    artifacts = {
        "tfidf": tfidf,
        "nn": nn,
        "id_to_idx": id_to_idx,
    }
    joblib_path = os.path.join(PROCESSED_DIR, "recommendation_model.joblib")
    joblib.dump(artifacts, joblib_path, compress=3)
    logging.info(f"Saved ML artifacts to {joblib_path}")


def precompute_aggregations(games_df: pd.DataFrame, reviews_df: pd.DataFrame):
    logging.info("Precomputing fast analytics aggregations...")

    # 1. Developer Stats (items per year, % free)
    dev_stats = {}
    for dev, group in games_df.groupby("developer"):
        if not dev or dev == "Unknown Developer":
            continue
        valid_years = group.dropna(subset=["year"])
        years_list = []
        for y, y_group in valid_years.groupby("year"):
            total_items = len(y_group)
            free_items = (y_group["price"] == 0).sum()
            pct_free = round((free_items / total_items) * 100, 2)
            years_list.append({
                "Año": int(y),
                "Cantidad de Items": int(total_items),
                "Contenido Free": f"{pct_free}%"
            })
        # Sort by year
        years_list.sort(key=lambda x: x["Año"], reverse=True)
        dev_stats[dev] = years_list

    with open(os.path.join(PROCESSED_DIR, "developer_stats.json"), "w", encoding="utf-8") as f:
        json.dump(dev_stats, f, ensure_ascii=False)

    # 2. Developer Sentiment Analysis
    # Merge reviews with games to map item_id -> developer
    game_dev_map = games_df.set_index("id")["developer"].to_dict()
    reviews_df["developer"] = reviews_df["item_id"].map(game_dev_map)

    dev_sentiment = {}
    for dev, group in reviews_df.dropna(subset=["developer"]).groupby("developer"):
        if not dev or dev == "Unknown Developer":
            continue
        counts = group["sentiment_analysis"].value_counts()
        dev_sentiment[dev] = {
            "Negative": int(counts.get(0, 0)),
            "Neutral": int(counts.get(1, 0)),
            "Positive": int(counts.get(2, 0))
        }

    with open(os.path.join(PROCESSED_DIR, "developer_sentiment.json"), "w", encoding="utf-8") as f:
        json.dump(dev_sentiment, f, ensure_ascii=False)

    # 3. Genre Rankings (by total catalog games & total user reviews)
    all_genres = []
    for g_list in games_df["genres"]:
        all_genres.extend(g_list)
    genre_series = pd.Series(all_genres).value_counts()
    genre_rankings = {genre: {"rank": rank + 1, "count": int(cnt)} for rank, (genre, cnt) in enumerate(genre_series.items())}

    with open(os.path.join(PROCESSED_DIR, "genre_rankings.json"), "w", encoding="utf-8") as f:
        json.dump(genre_rankings, f, ensure_ascii=False)

    # 4. User for Genre (top 5 users with most games reviewed for each genre)
    game_genres_map = games_df.set_index("id")["genres"].to_dict()
    reviews_df["genres"] = reviews_df["item_id"].map(game_genres_map)

    # Explode genres on reviews
    exploded_genre_reviews = reviews_df.dropna(subset=["genres"]).explode("genres")
    user_for_genre = {}
    for genre, group in exploded_genre_reviews.groupby("genres"):
        top_users = group["user_id"].value_counts().head(5)
        user_list = []
        for uid, count in top_users.items():
            u_url = group[group["user_id"] == uid]["user_url"].iloc[0]
            user_list.append({
                "user_id": uid,
                "user_url": u_url,
                "interactions_count": int(count)
            })
        user_for_genre[genre] = user_list

    with open(os.path.join(PROCESSED_DIR, "user_for_genre.json"), "w", encoding="utf-8") as f:
        json.dump(user_for_genre, f, ensure_ascii=False)

    # 5. User Data (money spent, recommendation percentage, item count)
    game_price_map = games_df.set_index("id")["price"].to_dict()
    reviews_df["price"] = reviews_df["item_id"].map(game_price_map).fillna(0.0)

    user_stats = {}
    for uid, group in reviews_df.groupby("user_id"):
        money_spent = float(group["price"].sum())
        total_items = len(group)
        rec_pct = float(group["recommend"].mean() * 100) if total_items > 0 else 0.0
        user_stats[uid] = {
            "user_id": uid,
            "money_spent_usd": round(money_spent, 2),
            "recommend_percentage": f"{round(rec_pct, 2)}%",
            "items_count": total_items
        }

    with open(os.path.join(PROCESSED_DIR, "user_stats.json"), "w", encoding="utf-8") as f:
        json.dump(user_stats, f, ensure_ascii=False)

    logging.info("Analytics aggregations successfully serialized.")


def main():
    games_path = os.path.join(DATA_DIR, "steam_games.parquet")
    reviews_path = os.path.join(DATA_DIR, "user_reviews.parquet")

    games_df = clean_games(games_path)
    reviews_df = clean_reviews_and_sentiment(reviews_path)

    # Save cleaned parquets
    games_out = os.path.join(PROCESSED_DIR, "games.parquet")
    reviews_out = os.path.join(PROCESSED_DIR, "reviews.parquet")
    games_df.to_parquet(games_out, index=False)
    reviews_df.to_parquet(reviews_out, index=False)
    logging.info(f"Saved cleaned datasets to {games_out} and {reviews_out}")

    # Train and export ML recommender
    train_and_export_recommender(games_df)

    # Precompute aggregations
    precompute_aggregations(games_df, reviews_df)
    logging.info("ETL, Feature Engineering and ML Pipeline completed successfully!")


if __name__ == "__main__":
    main()
