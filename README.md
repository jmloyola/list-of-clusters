# Clustering utilizando el algoritmo Lista de Clusters

Este repositorio contiene el trabajo realizado para el curso de posgrado _"Nuevas Propuestas Para Búsquedas Por Similitud En Bases De Datos Métricas"_ de la Universidad Nacional de San Luis (Argentina) en el año 2017.

En este repositorio se encuentran:
- Una implementación "naive" en Python del algoritmo [Lista de Clusters](https://ieeexplore.ieee.org/abstract/document/878182).
- Notebooks de Jupyter que muestran la aplicación del algoritmo para la tarea de clustering sobre conjuntos de datos sintéticos y se lo compara con algoritmos clásicos de clustering.
- Informe final.

## Dependencias

Para poder recrear las gráficas y ejecutar las notebooks de Jupyter es necesario instalar Python y los paquetes adecuados. Para ello recomendamos utilizar __conda__ como gestor de entornos. La forma más sencilla es instalando [Anaconda Python](https://www.anaconda.com/download/). Una vez instalado este, se puede ejecutar el siguiente comando en un terminal para generar el mismo ambiente de clustering utilizado para generar las gráficas: `conda env create -f environment.yml`.
Para activar el ambiente se debe utilizar el siguiente comando:
- En Windows, ejecuta: `activate clustering`.
- En macOS y Linux, ejecuta: `source activate clustering`.

Luego, para correr la notebook de Jupyter, ejecutar el comando: `jupyter notebook`.