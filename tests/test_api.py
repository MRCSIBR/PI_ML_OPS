import pytest
from fastapi.testclient import TestClient
from main import app

@pytest.fixture(scope="module")
def client():
    # Use context manager to trigger lifespan startup and shutdown
    with TestClient(app) as test_client:
        yield test_client

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "endpoints_disponibles" in data
    assert "version" in data

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["games_catalog_count"] > 20000
    assert data["users_count"] > 20000
    assert data["recommendation_model_loaded"] is True

def test_developer_stats_success(client):
    # Test with a developer known to be in Steam catalog
    response = client.get("/developer/Valve")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "Año" in data[0]
    assert "Cantidad de Items" in data[0]
    assert "Contenido Free" in data[0]

def test_developer_stats_not_found(client):
    response = client.get("/developer/NonExistentDev123XYZ")
    assert response.status_code == 404

def test_userdata_success(client):
    # Test with a known active user in the dataset
    response = client.get("/userdata/76561197970982479")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "76561197970982479"
    assert "dinero_gastado" in data
    assert "porcentaje_recomendacion" in data
    assert "cantidad_de_items" in data

def test_userdata_not_found(client):
    response = client.get("/userdata/unknown_user_99999999")
    assert response.status_code == 404

def test_countreviews_success(client):
    response = client.get("/countreviews?start_date=2011-01-01&end_date=2015-12-31")
    assert response.status_code == 200
    data = response.json()
    assert data["start_date"] == "2011-01-01"
    assert data["end_date"] == "2015-12-31"
    assert data["cantidad_usuarios"] > 0
    assert "%" in data["porcentaje_recomendacion"]

def test_genre_ranking(client):
    response = client.get("/genre/Action")
    assert response.status_code == 200
    data = response.json()
    assert data["genre"] == "Action"
    assert "puesto_ranking" in data
    assert data["puesto_ranking"] >= 1
    assert data["total_juegos"] > 0

def test_genre_not_found(client):
    response = client.get("/genre/NonExistentGenre999")
    assert response.status_code == 404

def test_user_for_genre(client):
    response = client.get("/userforgenre/Action")
    assert response.status_code == 200
    data = response.json()
    assert data["genre"] == "Action"
    assert "top_5_users" in data
    assert len(data["top_5_users"]) <= 5
    assert len(data["top_5_users"]) > 0
    assert "user_id" in data["top_5_users"][0]
    assert "user_url" in data["top_5_users"][0]

def test_sentiment_analysis_developer(client):
    response = client.get("/sentiment_analysis/Valve")
    assert response.status_code == 200
    data = response.json()
    assert "Negative" in data
    assert "Neutral" in data
    assert "Positive" in data
    assert (data["Negative"] + data["Neutral"] + data["Positive"]) > 0

def test_ml_game_recommendation_by_id(client):
    # Test recommendation for game ID 761140 (Lost Summoner Kitty)
    response = client.get("/recomendacion_juego/761140?top_n=5")
    assert response.status_code == 200
    data = response.json()
    assert data["item_id"] == "761140"
    assert "game_title" in data
    recs = data["recomendaciones"]
    assert len(recs) == 5
    for rec in recs:
        assert "item_id" in rec
        assert "title" in rec
        assert "similarity" in rec
        assert 0.0 <= rec["similarity"] <= 1.0

def test_ml_game_recommendation_by_name(client):
    response = client.get("/recomendacion_juego_por_nombre?title=Portal&top_n=5")
    assert response.status_code == 200
    data = response.json()
    assert "recomendaciones" in data
    assert len(data["recomendaciones"]) == 5

def test_ml_user_recommendation(client):
    response = client.get("/recomendacion_usuario/76561197970982479?top_n=5")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "76561197970982479"
    assert "recomendaciones" in data
    assert len(data["recomendaciones"]) > 0

def test_api_v1_prefix_compatibility(client):
    # Check that routes also work with /api/v1 prefix
    response = client.get("/api/v1/genre/Action")
    assert response.status_code == 200
    assert response.json()["genre"] == "Action"
