# Steam Games Recommendation System & MLOps API (Producción)

Sistema de recomendación de videojuegos de Steam impulsado por **Machine Learning** y API REST de alto rendimiento construida con **FastAPI**, lista para despliegue en producción (Docker, Render, Railway, AWS).

---

## 🚀 Características Principales

1. **Sistema de Recomendación por Machine Learning (Item-Item & User-Item)**:
   - Basado en filtrado por contenido utilizando **TF-IDF Vectorization** y **NearestNeighbors** con métrica de **Similitud del Coseno** sobre las características compuestas de los videojuegos (géneros, etiquetas temáticas y desarrollador).
   - Inferencia sub-milisegundo gracias a precomputación indexada y modelo serializado de respaldo para consultas dinámicas.
2. **Feature Engineering & NLP Sentiment Analysis**:
   - Clasificación de 59,305 reseñas de usuarios mediante procesamiento de lenguaje natural (NLP) en escala `0` (Negativo), `1` (Neutral / Sin reseña) y `2` (Positivo).
3. **Arquitectura Optimizada para Producción**:
   - Carga en memoria única durante el ciclo de vida del servidor (`lifespan`). Cero lecturas de disco por petición HTTP.
   - Tipado y validación estricta con modelos **Pydantic v2**.
   - Soporte CORS habilitado para integración con cualquier frontend.
   - Suite de pruebas automatizadas con **pytest**.
   - Contenedor Docker listo con healthcheck integrado.

---

## 📁 Estructura del Proyecto

```text
MLSteam/
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── endpoints/
│   │       │   ├── games.py            # /developer, /genre, /userforgenre
│   │       │   ├── users.py            # /userdata, /countreviews
│   │       │   ├── reviews.py          # /sentiment_analysis
│   │       │   └── recommendations.py  # /recomendacion_juego, /recomendacion_usuario
│   │       └── api.py                  # Router agregador v1
│   ├── core/
│   ├── models/
│   │   └── schemas.py                  # Modelos Pydantic v2
│   ├── services/
│   │   ├── data_service.py             # Singleton in-memory data loader
│   │   └── ml_service.py               # Motor de inferencia Machine Learning
│   └── config.py                       # Configuración y Settings
├── data/
│   ├── processed/                      # Datasets limpios y modelo serializado
│   ├── steam_games.parquet             # Dataset original
│   └── user_reviews.parquet            # Dataset original
├── scripts/
│   ├── build_dataset.py                # Pipeline ETL, NLP y entrenamiento ML
│   └── create_recommendation_notebook.py
├── tests/
│   └── test_api.py                     # Suite de pruebas con pytest
├── conftest.py
├── Dockerfile                          # Dockerfile de producción
├── docker-compose.yml                  # Orquestación con Docker Compose
├── main.py                             # Punto de entrada de la aplicación FastAPI
├── Model_Recomendacion.ipynb           # Notebook detallado del modelo de recomendación
├── requirements.txt                    # Dependencias
└── README.md
```

---

## 📓 Notebook del Modelo de Machine Learning

El repositorio incluye el notebook [`Model_Recomendacion.ipynb`](Model_Recomendacion.ipynb) con el análisis exhaustivo y didáctico del motor de recomendación:
- **Fundamentos Matemáticos**: Formulación rigurosa de TF-IDF y Similitud del Coseno.
- **Análisis de Eficiencia en Memoria**: Explicación de cómo se redujo el uso de memoria de 4.06 GB (matriz densa) a tan solo 2.32 MB (matriz esparsa comprimida), permitiendo su despliegue en contenedores gratuitos/micro en la nube.
- **Demostración Práctica**: Pruebas interactivas con casos reales de juegos populares (`Portal`, `Counter-Strike`, `Lost Summoner Kitty`) y recomendaciones personalizadas por usuario.


---

## 🛠️ Instalación y Uso Local

### 1. Clonar y crear entorno virtual
```bash
git clone https://github.com/MRCSIBR/PI_ML_OPS.git
cd PI_ML_OPS

python -m venv venv
# En Windows:
.\venv\Scripts\activate
# En Linux/macOS:
source venv/bin/activate
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. (Opcional) Re-ejecutar el pipeline de datos y entrenamiento ML
Los datos preprocesados ya vienen generados en `data/processed/`. Si deseas volver a correr el pipeline:
```bash
python scripts/build_dataset.py
```

### 4. Iniciar la API
```bash
uvicorn main:app --reload --port 8000
```
La API estará disponible en `http://localhost:8000` y la documentación interactiva Swagger en `http://localhost:8000/docs`.

---

## 🐳 Ejecución con Docker

### Con Docker Compose:
```bash
docker compose up --build -d
```

### Con Docker CLI:
```bash
docker build -t steam-mlops-api .
docker run -d -p 8000:8000 --name steam-api steam-mlops-api
```

---

## 🧪 Pruebas Automatizadas

Para ejecutar los 15 tests unitarios y de integración:
```bash
python -m pytest tests/ -v
```

---

## 📌 Documentación de Endpoints

### 1. Machine Learning: Recomendación Ítem-Ítem
- **Ruta**: `GET /recomendacion_juego/{item_id}`
- **Parámetros**: `item_id` (string), opcional: `top_n` (default=5).
- **Ejemplo**: `GET /recomendacion_juego/761140`
```json
{
  "item_id": "761140",
  "game_title": "Lost Summoner Kitty",
  "recomendaciones": [
    {
      "item_id": "563120",
      "title": "Desolate Wastes: Vendor Chronicles",
      "similarity": 0.8523,
      "genres": ["Action", "Indie", "RPG", "Simulation", "Strategy"],
      "price": 4.99
    },
    {
      "item_id": "361520",
      "title": "World of Cinema - Directors Cut",
      "similarity": 0.7915,
      "genres": ["Casual", "Indie", "Simulation", "Strategy"],
      "price": 9.99
    }
  ]
}
```

### 2. Machine Learning: Búsqueda y Recomendación por Título
- **Ruta**: `GET /recomendacion_juego_por_nombre?title={nombre}`
- **Ejemplo**: `GET /recomendacion_juego_por_nombre?title=Portal`

### 3. Machine Learning: Recomendación Usuario-Ítem
- **Ruta**: `GET /recomendacion_usuario/{user_id}`
- **Ejemplo**: `GET /recomendacion_usuario/76561197970982479`

### 4. Estadísticas de Desarrollador
- **Ruta**: `GET /developer/{desarrollador}`
- **Ejemplo**: `GET /developer/Valve`
```json
[
  {
    "Año": 2012,
    "Cantidad de Items": 2,
    "Contenido Free": "50.0%"
  },
  {
    "Año": 2011,
    "Cantidad de Items": 1,
    "Contenido Free": "0.0%"
  }
]
```

### 5. Datos de Usuario
- **Ruta**: `GET /userdata/{User_id}`
- **Ejemplo**: `GET /userdata/76561197970982479`
```json
{
  "user_id": "76561197970982479",
  "dinero_gastado": "29.98 USD",
  "porcentaje_recomendacion": "100.0%",
  "cantidad_de_items": 3
}
```

### 6. Conteo de Reseñas por Rango de Fechas
- **Ruta**: `GET /countreviews?start_date=YYYY-MM-DD&end_date=YYYY-MM-DD`
- **Ejemplo**: `GET /countreviews?start_date=2011-01-01&end_date=2015-12-31`
```json
{
  "start_date": "2011-01-01",
  "end_date": "2015-12-31",
  "cantidad_usuarios": 16938,
  "porcentaje_recomendacion": "89.62%"
}
```

### 7. Ranking de Géneros
- **Ruta**: `GET /genre/{genre}`
- **Ejemplo**: `GET /genre/Action`
```json
{
  "genre": "Action",
  "puesto_ranking": 2,
  "total_juegos": 11321
}
```

### 8. Usuarios Destacados por Género
- **Ruta**: `GET /userforgenre/{genre}`
- **Ejemplo**: `GET /userforgenre/Action`
```json
{
  "genre": "Action",
  "top_5_users": [
    {
      "user_id": "Spik3",
      "user_url": "http://steamcommunity.com/id/Spik3",
      "interactions_count": 22
    }
  ]
}
```

### 9. Análisis de Sentimiento por Desarrollador
- **Ruta**: `GET /sentiment_analysis/{empresa_desarrolladora}`
- **Ejemplo**: `GET /sentiment_analysis/Valve`
```json
{
  "desarrollador": "Valve",
  "Negative": 1289,
  "Neutral": 4124,
  "Positive": 4170
}
```

### 10. Health Check
- **Ruta**: `GET /health`
```json
{
  "status": "healthy",
  "games_catalog_count": 22529,
  "users_count": 25485,
  "recommendation_model_loaded": true
}
```

---

## 🌐 Despliegue en la Nube (Render / Railway)

1. Conectar este repositorio en el panel de **Render** o **Railway**.
2. Seleccionar entorno **Python** o **Docker**.
   - Si se usa **Python**:
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - Si se usa **Docker**:
     - El servicio detectará el `Dockerfile` automáticamente.
3. El endpoint de health check estará en `/health`.
