
# Técnicas de aprendizaje automático aplicadas en la predicción de potenciales compradores en un e-commerce de productos empresariales en Colombia

![e_commerce](https://github.com/jhontd03/PredictionCustomer/blob/master/img/e_commerce.png "e_commerce")

## [Contenido](#Contenido)

- [Resumen](#Resumen)
- [Instalación](#Instalación)
- [Requisitos](#Requisitos)
- [Uso](#Uso)
- [Estructura del repositorio](#Estructura-del-repositorio)
- [Resultados](#Resultados)
- [Licencia](#Licencia)
- [Autor](#Autor)

## Resumen

En el presente repositorio se presenta la fase de diseño e implementación del trabajo de fin de máster titulado: **Técnicas de aprendizaje automático aplicadas en la predicción de potenciales compradores en un e-commerce de productos empresariales en Colombia**, que se enfoca en un desafío fundamental en el entorno del comercio electrónico: la predicción de potenciales compradores en un e-commerce de productos empresariales en Colombia. En un contexto donde la información empresarial es esencial, la capacidad de anticipar quiénes se convertirán en clientes es crucial para la estrategia de negocio.
Para abordar este desafío, se aplicaron y evaluaron diversas técnicas de aprendizaje automático y se adoptó la metodología CRISP-DM para su desarrollo, abordando únicamente cinco de las seis fases, que se describen a continuación. 1) Comprender los retos comerciales y definir objetivos de predicción y retención, 2) analizar la calidad de los datos considerando diversas fuentes, 2) Preparar los datos mediante limpieza y transformación, 4) Implementar técnicas de aprendizaje automático para construir modelos predictivos y 5) Evaluar el rendimiento y comparar la efectividad de los modelos.
Los resultados obtenidos demuestran una exploración en profundidad de las técnicas y modelos de aprendizaje automático aplicados a la predicción de compradores, ofreciendo así una contribución significativa al campo de la ciencia de datos en el comercio electrónico. Adicionalmente, este trabajo no solo proporciona una aproximación analítica, sino que además demuestra como un enfoque basado en datos y técnicas avanzadas de análisis, puede contribuir a las empresas en la toma de decisiones y promover mejores estrategias de crecimiento en el mercado digital.

## Instalación

## Requisitos

Para la ejecución de la aplicación es necesario instalar  [python 3.9.16](https://www.python.org/downloads/release/python-3916/) y los paquetes incluidos en el archivo [requeriments.txt](https://github.com/jhontd03/accidentapp/blob/master/requirements.txt). 
Para usuarios de windows se recomienda usar el emulador de la terminal de comandos [cmder](https://cmder.app/)  similar al bash de linux. 

En caso que el archivo python.exe no se puede ejectuar desde la terminal de comandos, es necesario que los usuarios de windows agreguen el directorio de instalación de python a las variables de entorno, tal como se describe
en el siguiente [tutorial.](https://realpython.com/add-python-to-path/)

## Uso

Clone el presente repositorio cree un entorno virtual, instale los paquetes y ejecute el código python directamente.

```
git clone https://github.com/jhontd03/PredictionCustomer.git
cd PredictionCustomer
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter notebook
```

## Estructura del repositorio

El árbol de directorios del repositorio es el siguiente:
```
.
│   1_etl_eda.ipynb
│   2_eda_dataset.ipynb
│   3_model_pred.ipynb
│   etl_process.py
│   model_prediction.py
│   LICENSE
│   README.md
│   __init__.py
│
├───data
│   │   Linea1-Consumos.txt
│   │   Linea1-Empresas_Consultadas.txt
│   │   Linea1-Primer_Consumo.txt
│   │   Linea1-Sesiones.txt
│   │   Linea1-Usuarios.txt
│   │   Linea1-Ventas.txt
│   │
│   ├───models
│   │   
│   └───reports
│
├───img
│       e_commerce.jpg
│
└───reports
        1_etl_eda.html
        2_eda_dataset.html
        3_model_pred.html

```
## Resultados

A continuación se describe un resumen del contenido de los notebooks y scripts implementados.

### Notebook 1. Proceso de extracción, transformación y análisis exploratorio de datos

En este notebook se desarrolla la primera parte de la fase de diseño e implementación del trabajo de fin de máster. En la sección 2, se realiza el cargue de los seis dataframe, posteriomente en la sección 3, se verifica y transforma el tipo de variable/atributo, teniendo en cuenta la información suministrada por el director del trabajo de fin de máster y un análisis de detallado de cada dataframe. En la sección 4, sobre los dataframe sesiones y consumos se aplica una operación de agregación/agrupación de datos usando como clave el atributo id_usuario. Dicha operación se puede considerar como una tarea de ingeniería de características, dado que se crearon nuevos atributos. En la sección 5, se realiza un estudio y procesamiento de valores nulos, aplicando diferentes técnicas de imputación en función del tipo de variable/atributo. En la sección 6 se realiza un estudio basado en estadística descriptiva, gráficos de barras y cajas, para determinar las caracteristicas más relevantes de cada atributo/variable de los seis dataframe. Así mismo, con base en los gráficos se identifican la presencia de valores atípicos. En la sección 7 se concatena los dataframe, para conformar el conjunto de datos final, que se usará para la generación de los modelos predictivos. Finalmente, en la sección 8 se presenta las conclusiones más relevantes del estudio realizado en el presente notebook.

### Notebook 2. Análisis exploratorio de datos

En este notebook se desarrolla la segunda fase del diseño e implementación del trabajo de fin de máster. En la Sección 2, se realiza un análisis descriptivo detallado de las variables del dataset. Este análisis se realiza de manera secuencial, iniciando con las variables numéricas y seguido por las variables categóricas, empleando enfoques tanto univariados como multivariados. La Sección 3 está dedicada a la creación del dataset final, el cual será la base para el desarrollo de modelos predictivos. Finalmente, la Sección 4 recoge las conclusiones más significativas derivadas del estudio realizado en este notebook

### Notebook 3. Selección y evaluación de modelos predictivos

En este notebook se desarrolla la tercera fase del diseño e implementación del trabajo de fin de máster. En la sección 2 se realiza el proceso de selección de características. Para ello se determina el mejor modelo de ensamble de referencia (que tienen integrado funcionalidades de ranking de características) y sobre este se aplica la técnica RFE (Recursive Feature Elimination), para seleccionar las variables más relevantes. En la sección 3 se desarrolla el primer método para la generación de modelos predictivos. Dicho método se conoce como muestreo, el cual tiene diferentes técnicas que permite balancear las clases de la variable objetivo. Para determinar las mejores técnicas, en primer lugar, se seleccionan los mejores modelos de clasificación. Luego a los mejores modelos se les aplica diferentes métodos de muestreo. Sobre los mejores modelos obtenidos se optimiza hiperparámetros, entrenan y evalúan. En la sección 4 se desarrolla el segundo método de generación de modelos predictivos basado en ensamble. Para ello se determinan los mejores modelos y posteriormente se optimiza hiperparámetros, entrenan y evalúan. En la sección 5 se aplica el tercer y último método para generar modelos predictivos que se basa en el concepto de Cost-sensitive learning (CSL). En este caso se usaron los modelos de ensamble que obtuvieron el mejor desempeño en la sección 2 y que tienen integrado la funcionalidad (CSL). Posteriormente en la sección 6, se generan un modelo final basado en la técnica de voting, donde se incluyen los modelos que mejor se desempeñaron en las secciones anteriores. En la sección 7 se hace un consolidado de las métricas más relevantes de los modelos obtenidos y su respectivo análisis. Finalmente, la Sección 9 recoge las conclusiones más significativas derivadas del estudio realizado en este notebook.

### Script 1. etl_process.py

Contiene funciones para el preprocesamiento y análisis exploratorio de datos, donde se incluyen funciones para el tratamiento de valores nulos, transformaciones de datos, normalización,
análisis descriptivo, y visualizaciones para facilitar la comprensión de las características  del conjunto de datos.

### Script 2. model_prediction.py

Contiene funciones para la evaluación y comparación de modelos de aprendizaje automático en un contexto de clasificación. Las funciones proporcionan herramientas detalladas para analizar el rendimiento de los modelos a través de varias métricas, incluyendo precisión, recall, curvas ROC y Precision-Recall, así como gráficos informativos.

## Licencia

El código del presente proyecto se distribuye bajo licencia [MIT](https://github.com/jhontd03/accidentapp/blob/master/LICENSE). Por la presente se concede permiso, de forma gratuita, a cualquier persona que obtenga una copia de este software y de los archivos de documentación asociados (el "Software"), para comerciar con el Software sin restricciones, incluyendo sin limitación los derechos de uso, copia, modificación, fusión, publicación, distribución, sublicencia y/o venta de copias del Software.

## Autor

Jhon Jairo Realpe

jrealpe0@uoc.edu