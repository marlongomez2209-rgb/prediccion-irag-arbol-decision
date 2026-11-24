# Predicción de atenciones por IRAG mediante árboles de decisión

## Descripción
Este proyecto consiste en el desarrollo de una aplicación web para estimar las atenciones semanales por Infección Respiratoria Aguda Grave (IRAG) en Colombia, utilizando técnicas de inteligencia artificial y aprendizaje automático.

Se implementó un modelo de árbol de decisión entrenado con datos históricos de vigilancia epidemiológica.

## Objetivo
Desarrollar un modelo predictivo que permita estimar el comportamiento semanal de las atenciones por IRAG a partir de registros históricos.

## Tecnologías utilizadas
- Python
- Google Colab
- Pandas
- Scikit-learn
- Joblib
- Streamlit

## Variables utilizadas
El modelo utiliza seis variables predictoras:

1. Año.
2. Semana epidemiológica.
3. Atenciones de la semana anterior.
4. Promedio de atenciones de las últimas 4 semanas.
5. Promedio de atenciones de las últimas 8 semanas.
6. Atenciones de hace 52 semanas.

## Modelo de inteligencia artificial
Se utilizó un algoritmo de árbol de decisión para realizar las predicciones.

Los mejores parámetros obtenidos durante la optimización fueron:

- Profundidad máxima: 4.
- Mínimo de muestras por hoja: 20.
- Mínimo de muestras para dividir un nodo: 2.

## Resultados de evaluación

| Métrica | Resultado |
|---|---:|
| MA
