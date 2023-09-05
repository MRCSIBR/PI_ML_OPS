# PI_ML_OPS
Proyecto individual 1

En nuestro rol como data scientist para Steam debemos crear un sistema de recomendacion de videojuegos para la empresa. Debemos limpiar los datasets haciendo un trabajo rapido de data engineering. Lograr tener un MVP (minimum viable product).
## Estructura del repositorio:
            |
            |__* Data:       Carpeta con datasets
            |
            |__* Notebooks:  LOAD_steam_games.ipynb
            |                DesarrolloAPI.ipynb
            |                ETL_EDA_Steam.ipynb
            |
            |__* Codigo:     main.py
                             ML_boilerplate.py

## Preprocesamiento de data

Decidi usar `el formato de archivo '.parquet' que es mas eficiente en big data` y permite trabajar de forma agil, para convertir 
los '.json' a '.parquet' instale los siguientes modulos: 

             !pip install pyarrow
             !pip install fastparquet

En el notebook 'LOAD_steam_games.ipynb' esta documentado el proceso.

## ETL EDA

Luego de realizar los procesos de EDA ETL para entender los datos provistos continue con el desarrollo de la API
con los siguientes endpoints: 

  `games`
  `userdata`
  `countreviews`
  `genre`
  `userforgenre`
  `developer`
  `sentiment_analysis`

  El proyecto incluye un notebook y datasets usados

  ## DEPLOY en RENDER

  **`~ El deploy funciona de forma local pero no logra completar el build
  en Render porque la plataforma solo soporta hasta python 3.7`**
  
  Para el despliegue de la API se uso Render
  disponible en el siguiente
  link: https://deploy-api-mlops-zm86.onrender.com

## TODO:

- Deploy de FastAPI con docker
- Implementar machine learning

