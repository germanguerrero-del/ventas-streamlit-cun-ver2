"""
ACA N.2 - Aprendizaje Automatico (CUN) | App interactiva en Streamlit
Pilar: Ventas. Permite explorar los datos, consultar las bases SQL, elegir
entre las tres arquitecturas entrenadas (CNN, Secuencial LSTM+GRU,
Transformer), ver sus metricas/curvas y generar el pronostico del dia
siguiente, ademas de descargar el poster en PDF.

Ejecucion local:    streamlit run app.py
Red local:          streamlit run app.py --server.address 0.0.0.0
Tunel publico:       python iniciar_ngrok.py   (o celda de ngrok del notebook)
"""
import json
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

import pipeline as pl

st.set_page_config(page_title="ACA 2 · Ventas · CNN/RNN/Transformer", layout="wide")

RUTA_PESOS = Path("pesos")
RUTA_SQL = Path("bases_sql")
RUTA_DATOS = Path("train.csv") if Path("train.csv").exists() else Path("datos/train.csv")


@st.cache_data(show_spinner=False)
def cargar_datos():
    df = pl.read_sales(str(RUTA_DATOS))
    daily = pl.daily_sales(df)
    return df, daily


@st.cache_data(show_spinner=False)
def cargar_metricas(tipo: str):
    ruta = RUTA_PESOS / f"metricas_{tipo}.json"
    if not ruta.exists():
        return None
    with open(ruta, encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def cargar_historial(tipo: str):
    ruta = RUTA_PESOS / f"historial_{tipo}.csv"
    return pd.read_csv(ruta) if ruta.exists() else None


@st.cache_data(show_spinner=False)
def cargar_predicciones(tipo: str):
    ruta = RUTA_PESOS / f"predicciones_{tipo}.csv"
    return pd.read_csv(ruta) if ruta.exists() else None


@st.cache_resource(show_spinner=False)
def cargar_modelo(tipo: str):
    return pl.cargar_modelo(tipo)


def modelos_disponibles():
    return [t for t in pl.MODELOS if (RUTA_PESOS / f"modelo_{t}.pt").exists()]


st.title("ACA N.° 2 · Aprendizaje Automático aplicado a Ventas")
st.caption("CUN · Especialización en Inteligencia Artificial · Docente: Hamilton Rivera Flor")

disponibles = modelos_disponibles()
if not disponibles:
    st.error(
        "No se encontraron pesos entrenados en `pesos/`. Ejecute primero el "
        "notebook `ACA2_Ventas_Multired_Notebook.ipynb` (o `python pipeline.py`) "
        "para entrenar y persistir las tres arquitecturas."
    )
    st.stop()

tabs = st.tabs(["\U0001F4CA EDA", "\U0001F5C4️ Bases SQL", "\U0001F9E0 Modelos y métricas",
                 "\U0001F52E Pronóstico interactivo", "\U0001F4C4 Póster PDF"])

# ---------------------------------------------------------------- EDA ----
with tabs[0]:
    df, daily = cargar_datos()
    c1, c2, c3 = st.columns(3)
    c1.metric("Transacciones", f"{len(df):,}")
    c2.metric("Días (serie diaria)", f"{len(daily):,}")
    c3.metric("Ventas totales (USD)", f"{df['Sales'].sum():,.0f}")

    st.subheader("Ventas diarias")
    st.line_chart(daily.set_index("date")["sales"])

    st.subheader("Ventas mensuales")
    mensual = daily.assign(mes=daily["date"].dt.to_period("M").astype(str)).groupby("mes")["sales"].sum()
    st.bar_chart(mensual)

    st.subheader("Ventas por categoría")
    cat = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
    st.bar_chart(cat)

    with st.expander("Vista de los datos crudos"):
        st.dataframe(df.head(200))

# --------------------------------------------------------------- SQL -----
with tabs[1]:
    st.write("Las tres bases SQLite generadas por el pipeline (PASO 1):")
    bases = {"transacciones": RUTA_SQL / "transacciones.sqlite",
             "analitica": RUTA_SQL / "analitica.sqlite",
             "predicciones": RUTA_SQL / "predicciones.sqlite"}
    base_sel = st.selectbox("Base de datos", list(bases.keys()))
    ruta_db = bases[base_sel]
    if ruta_db.exists():
        with sqlite3.connect(ruta_db) as conn:
            tablas = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table'", conn)["name"].tolist()
            tabla_sel = st.selectbox("Tabla", tablas)
            consulta = st.text_area("Consulta SQL", f"SELECT * FROM {tabla_sel} LIMIT 50")
            if st.button("Ejecutar consulta"):
                try:
                    st.dataframe(pd.read_sql_query(consulta, conn))
                except Exception as e:
                    st.error(f"Error en la consulta: {e}")
    else:
        st.warning("Esta base aún no existe. Ejecute el pipeline primero.")

# ---------------------------------------------------- Modelos y metricas -
with tabs[2]:
    comp_path = RUTA_PESOS / "comparacion_modelos.csv"
    if comp_path.exists():
        st.subheader("Comparación de las tres arquitecturas (partición de prueba)")
        comp = pd.read_csv(comp_path, index_col=0)
        comp.index = [pl.NOMBRES_MODELOS.get(i, i) for i in comp.index]
        st.dataframe(comp.style.format("{:.3f}"))

    tipo = st.selectbox("Arquitectura", disponibles, format_func=lambda t: pl.NOMBRES_MODELOS[t])
    metr = cargar_metricas(tipo)
    hist = cargar_historial(tipo)
    pred = cargar_predicciones(tipo)

    if metr:
        st.markdown(f"**Parámetros entrenables:** {metr['n_parametros']:,} &nbsp;|&nbsp; "
                     f"**Umbral de venta alta:** {metr['umbral']:.1f} USD &nbsp;|&nbsp; "
                     f"**Pronóstico próximo día:** {metr['pronostico_dia_siguiente']:.1f} USD")
        cols = st.columns(4)
        m_test = metr["metrics"]["test"]
        cols[0].metric("MAE (test)", f"{m_test['MAE']:.1f}")
        cols[1].metric("RMSE (test)", f"{m_test['RMSE']:.1f}")
        cols[2].metric("sMAPE (test)", f"{m_test['sMAPE']:.1f}%")
        cols[3].metric("R² (test)", f"{m_test['R2']:.3f}")
        cols2 = st.columns(2)
        cols2[0].metric("Accuracy venta alta", f"{m_test['Accuracy_alta_venta']:.3f}")
        cols2[1].metric("F1 venta alta", f"{m_test['F1_alta_venta']:.3f}")

    if hist is not None:
        st.subheader("Curvas de entrenamiento / validación")
        st.line_chart(hist.set_index("epoch")[["train_loss", "val_loss"]])

    if pred is not None:
        st.subheader("Real vs. predicho — partición de prueba")
        test_pred = pred[pred["particion"] == "test"].copy()
        test_pred["fecha"] = pd.to_datetime(test_pred["fecha"])
        st.line_chart(test_pred.set_index("fecha")[["real", "prediccion"]])

    if metr:
        st.subheader("Matriz de confusión auxiliar (venta alta vs. no alta) — test")
        cm = metr["metrics"]["test"]["confusion"]
        st.table(pd.DataFrame(cm, index=["Real: No alta", "Real: Alta"],
                               columns=["Pred: No alta", "Pred: Alta"]))

# --------------------------------------------------- Pronostico en vivo --
with tabs[3]:
    st.write("Genere el pronóstico de ventas del día siguiente con la arquitectura elegida, "
              "o cargue un CSV propio con una columna `sales` (al menos 28 valores diarios).")
    tipo2 = st.selectbox("Arquitectura para inferencia", disponibles,
                          format_func=lambda t: pl.NOMBRES_MODELOS[t], key="infer")
    modelo = cargar_modelo(tipo2)

    _, daily = cargar_datos()
    archivo = st.file_uploader("CSV opcional con columna 'sales' (si no se sube, se usa la serie histórica)",
                                type=["csv"])
    if archivo is not None:
        serie_ext = pd.read_csv(archivo)
        if "sales" not in serie_ext.columns:
            st.error("El CSV debe tener una columna llamada 'sales'.")
        else:
            daily_usuario = serie_ext.tail(max(len(serie_ext), pl.WINDOW + 1)).reset_index(drop=True)
            daily_usuario["date"] = pd.date_range(end=pd.Timestamp.today(), periods=len(daily_usuario))
            bundle_usr = pl.prepare(daily_usuario, window=pl.WINDOW)
            pronostico = pl.forecast_next(modelo, bundle_usr)
            st.success(f"Pronóstico ({pl.NOMBRES_MODELOS[tipo2]}) con datos cargados: **{pronostico:,.2f} USD**")
    else:
        bundle = pl.prepare(daily, window=pl.WINDOW)
        pronostico = pl.forecast_next(modelo, bundle)
        st.success(f"Pronóstico de ventas para el próximo día ({pl.NOMBRES_MODELOS[tipo2]}): "
                    f"**{pronostico:,.2f} USD**")
        st.caption(f"Basado en los últimos {pl.WINDOW} días de la serie histórica (train.csv).")

# -------------------------------------------------------------- Poster ---
with tabs[4]:
    st.write("Genere el póster tipo infografía en PDF exigido por el entregable oficial (Sección 4.2).")
    integrantes = st.text_input("Nombres de los integrantes (separados por coma)", "Completar antes de entregar")
    app_url = st.text_input("Enlace público de la app (ngrok / Streamlit Cloud)", "URL: pendiente de despliegue")
    if st.button("Generar póster PDF"):
        from pipeline import run as pipeline_run  # noqa: F401  (solo para confirmar import valido)
        info_poster = {"resultados": {t: {"metrics": cargar_metricas(t)["metrics"]} for t in disponibles}}
        from poster import make_poster
        ruta_pdf = make_poster(info_poster, integrantes=integrantes, app_url=app_url,
                                output="docs/Poster_ACA2_Ventas.pdf")
        st.success(f"Póster generado en: {ruta_pdf}")
        with open(ruta_pdf, "rb") as fh:
            st.download_button("Descargar póster PDF", fh, file_name="Poster_ACA2_Ventas.pdf")

st.sidebar.header("Acerca del proyecto")
st.sidebar.write(
    "Este tablero integra tres arquitecturas de redes neuronales profundas "
    "entrenadas sobre la misma serie de ventas diarias: **CNN (Conv1D)**, "
    "**Secuencial (LSTM+GRU)** y **Transformer**. Todas comparten el mismo "
    "preprocesamiento (ventanas de 28 días, normalización y aumento de "
    "datos) para una comparación justa."
)
st.sidebar.markdown("---")
st.sidebar.caption("Despliegue: local, red local o túnel público con ngrok "
                    "(ver `iniciar_ngrok.py` o la celda correspondiente del notebook).")
