
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Predicción de atenciones IRAG",
    page_icon="🏥",
    layout="centered"
)

@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_arbol_decision_irag (2).pkl")


@st.cache_data
def cargar_datos():

    archivo = "5._Datos_colectivos_de_infección_respiratoria_aguda_de_vigilancia_epidemiológica_20261123.csv"

    df = pd.read_csv(archivo)

    columnas = [
        "AÑO",
        "SEMANA EPIDEMIOLOGICA",
        "TOTAL DE EVENTOS DE MORBILIDAD DE IRAG POR CONSULTA EXTERNA Y URGENCIAS"
    ]

    datos = df[columnas].copy()

    datos.columns = [
        "anio",
        "semana_epidemiologica",
        "atenciones_irag"
    ]

    for columna in datos.columns:
        datos[columna] = pd.to_numeric(
            datos[columna], errors="coerce"
        )

    datos = datos.dropna()

    semanal = (
        datos.groupby(
            ["anio", "semana_epidemiologica"],
            as_index=False
        )["atenciones_irag"].sum()
    )

    semanal = semanal.sort_values(
        ["anio", "semana_epidemiologica"]
    ).reset_index(drop=True)

    semanal["semana_anterior"] = (
        semanal["atenciones_irag"].shift(1)
    )

    semanal["promedio_4_semanas"] = (
        semanal["atenciones_irag"].shift(1).rolling(4).mean()
    )

    semanal["promedio_8_semanas"] = (
        semanal["atenciones_irag"].shift(1).rolling(8).mean()
    )

    semanal["atenciones_hace_52_semanas"] = (
        semanal["atenciones_irag"].shift(52)
    )

    return semanal.dropna().reset_index(drop=True)


modelo = cargar_modelo()
datos = cargar_datos()

st.title("Predicción de atenciones por IRAG")
st.subheader("Bucaramanga, Colombia")

st.write("""
Esta aplicación utiliza un modelo de árbol de decisión
para estimar las atenciones semanales por infección
respiratoria aguda grave (IRAG), a partir de registros
históricos de vigilancia epidemiológica.
""")

st.divider()

st.header("Consultar predicción")

st.write(
    "Selecciona el año y la semana epidemiológica que deseas consultar."
)

anios = sorted(datos["anio"].unique())

anio_seleccionado = st.selectbox(
    "Año",
    anios,
    index=len(anios) - 1,
    format_func=lambda x: str(int(x))
)

semanas = sorted(
    datos.loc[
        datos["anio"] == anio_seleccionado,
        "semana_epidemiologica"
    ].unique()
)

semana_seleccionada = st.selectbox(
    "Semana epidemiológica",
    semanas,
    format_func=lambda x: str(int(x))
)

if st.button("Realizar predicción"):

    registro = datos[
        (datos["anio"] == anio_seleccionado) &
        (datos["semana_epidemiologica"] == semana_seleccionada)
    ]

    variables = [
        "anio",
        "semana_epidemiologica",
        "semana_anterior",
        "promedio_4_semanas",
        "promedio_8_semanas",
        "atenciones_hace_52_semanas"
    ]

    prediccion = modelo.predict(registro[variables])[0]

    st.divider()
    st.subheader("Resultado")

    st.metric(
        "Atenciones estimadas por IRAG",
        f"{max(0, round(prediccion)):,}"
    )

    st.caption(
        f"Año {int(anio_seleccionado)} · "
        f"Semana epidemiológica {int(semana_seleccionada)}"
    )

    st.info(
        "La cifra es una estimación del modelo y no representa "
        "el número de personas únicas enfermas."
    )

st.divider()

st.caption("Proyecto académico de Ingeniería Mecatrónica.")
