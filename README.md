# PI_ML_OPS
Proyecto individual 1

En mi rol como data scientist para Steam tuve la tarea de crear un sistema de recomendacion de videojuegos para la empresa. 
El repositorio contiene los notebooks que documentan el proceso realizado de ETL(Extraccion transformacion y carga de datos. Data Engineering (Conversion de .json a .parquet) etc. 
Con el objetivo de lograr tener un MVP (minimum viable product).

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

  **`~ Corregido: para que funcione el build hay que cambiar la variable PYTHON_VERSION a 3.11.2`**
  
  Para el despliegue de la API utilize la plataforma Render,
  disponible en el siguiente link: 
  https://mdi-mlops.onrender.com/docs

## TODO:

- ~~Deploy de FastAPI con docker~~
- Implementar modelo machine learning

