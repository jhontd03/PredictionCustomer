# Prediction Custome.

| Título del trabajo                                 | Técnicas de aprendizaje automático aplicadas en la predicción de potenciales compradores en un e-commerce de productos empresariales en Colombia. |
|---------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| Nombre del autor/a                                | Jhon Jairo Realpe                                                                                                             |
| Nombre del Tutor/a de TF                          | Santiago Rojo Muñoz                                                                                                           |
| Nombre del/de la PRA                             | Albert Solé-Ribalta                                                                                                           |
| Fecha de entrega                                  | ene-24                                                                                                                        |
| Titulación o programa                             | Máster Universitario en Ciencia de Datos                                                                                      |
| Área del Trabajo Final                            | Analítica Empresarial                                                                                                         |
| Idioma del trabajo                                | Castellano                                                                                                                    |
| Palabras clave                                    | Aprendizaje automático, predicción de compradores, e-commerce.                                                                |


## Resumen

En el presente repositorio se presenta la fase de diseño e implementación asociada al trabajo de fin de máster titulado: **Técnicas de aprendizaje automático aplicadas en la predicción de potenciales compradores en un e-commerce de productos empresariales en Colombia**, que se enfoca en un desafío fundamental en el entorno del comercio electrónico: la predicción de potenciales compradores en un e-commerce de productos empresariales en Colombia. En un contexto donde la información empresarial es esencial, la capacidad de anticipar quiénes se convertirán en clientes es crucial para la estrategia de negocio.
Para abordar este desafío, se aplicaron y evaluaron diversas técnicas de aprendizaje automático y se adoptó la metodología CRISP-DM para su desarrollo, abordando únicamente cinco de las seis fases, que se describen a continuación. 1) Comprender los retos comerciales y definir objetivos de predicción y retención, 2) analizar la calidad de los datos considerando diversas fuentes, 2) Preparar los datos mediante limpieza y transformación, 4) Implementar técnicas de aprendizaje automático para construir modelos predictivos y 5) Evaluar el rendimiento y comparar la efectividad de los modelos.
Los resultados obtenidos demuestran una exploración en profundidad de las técnicas y modelos de aprendizaje automático aplicados a la predicción de compradores, ofreciendo así una contribución significativa al campo de la ciencia de datos en el comercio electrónico. Adicionalmente, este trabajo no solo proporciona una aproximación analítica, sino que además demuestra como un enfoque basado en datos y técnicas avanzadas de análisis, puede contribuir a las empresas en la toma de decisiones y promover mejores estrategias de crecimiento en el mercado digital.


## Notebook 1. Proceso de extracción, transformación y análisis exploratorio de datos.

En este notebook se desarrolla la primera parte de la fase de diseño e implementación del trabajo de fin de máster. En la sección 2, se realiza el cargue de los seis dataframe, posteriomente en la sección 3, se verifica y transforma el tipo de variable/atributo, teniendo en cuenta la información suministrada por el director del trabajo de fin de máster y un análisis de detallado de cada dataframe. En la sección 4, sobre los dataframe sesiones y consumos se aplica una operación de agregación/agrupación de datos usando como clave el atributo id_usuario. Dicha operación se puede considerar como una tarea de ingeniería de características, dado que se crearon nuevos atributos. En la sección 5, se realiza un estudio y procesamiento de valores nulos, aplicando diferentes técnicas de imputación en función del tipo de variable/atributo. En la sección 6 se realiza un estudio basado en estadística descriptiva, gráficos de barras y cajas, para determinar las caracteristicas más relevantes de cada atributo/variable de los seis dataframe. Así mismo, con base en los gráficos se identifican la presencia de valores atípicos. En la sección 7 se concatena los dataframe, para conformar el conjunto de datos final, que se usará para la generación de los modelos predictivos. Finalmente, en la sección 8 se presenta las conclusiones más relevantes del estudio realizado en el presente notebook.



## Notebook 2. Análisis exploratorio de datos.

En este notebook se desarrolla la segunda fase del diseño e implementación del trabajo de fin de máster. En la Sección 2, se procede con un análisis descriptivo detallado de las variables del dataset. Este análisis se realiza de manera secuencial, iniciando con las variables numéricas y seguido por las variables categóricas, empleando enfoques tanto univariados como multivariados. La Sección 3 está dedicada a la creación del dataset final, el cual será la base para el desarrollo de modelos predictivos. Finalmente, la Sección 4 recoge las conclusiones más significativas derivadas del estudio realizado en este notebook

