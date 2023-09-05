# PI_ML_OPS
Proyecto individual 1

En nuestro rol como data scientist para Steam debemos crear un sistema de recomendacion de videojuegos para la empresa. Debemos limpiar los datasets haciendo un trabajo rapido de data engineering. Tener un MVP antes de la fecha de cierre.
## Estructura del repositorio:
            |
            |__* Data:       Carpeta con datasets
            |
            |__* Notebooks : DesarrolloAPI.ipynb
            |                ETL_EDA_Steam.ipynb
            |
            |__* Codigo:     main.py
                             ML_boilerplate.py

## Preprocesamiento de data

Decidi usar el formato parquet que ocupa menos espacio y permite trabajar de forma mas eficiente, para convertir 
los .json a .parquet instale los siguientes modulos: 
            `pip install pyarrow`
            `pip install fastparquet`

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

  Para el despliegue de la API se uso Render
  disponible en el siguiente
  link: https://deploy-api-mlops-zm86.onrender.com

## TODO:

Implementar machine learning
