"""Genera el poster tipo infografia (PDF) exigido por la Seccion 4 del ACA N.2.
Requiere reportlab. Resume: integrantes, objetivo, problema, dataset, flujo
metodologico, las TRES arquitecturas de red neuronal comparadas, metricas
principales y enlace de la aplicacion.
"""
from pathlib import Path
from reportlab.lib.pagesizes import A3
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.pdfgen import canvas

from pipeline import NOMBRES_MODELOS


def _parrafo(c, texto, x, y, ancho, tam=10, lead=13, negrita=False):
    c.setFont("Helvetica-Bold" if negrita else "Helvetica", tam)
    import textwrap
    max_chars = max(10, int(ancho / (tam * 0.5)))
    lineas = []
    for parr in texto.split("\n"):
        lineas += textwrap.wrap(parr, max_chars) or [""]
    for linea in lineas:
        c.drawString(x, y, linea)
        y -= lead
    return y


def make_poster(info: dict, integrantes: str, app_url: str,
                 output: str = "docs/Poster_ACA2_Ventas.pdf",
                 video_url: str = "URL del video: pendiente de grabacion"):
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    W, H = A3
    c = canvas.Canvas(str(output), pagesize=A3)

    # Encabezado
    c.setFillColor(colors.HexColor("#102A43"))
    c.rect(0, H - 3.2 * cm, W, 3.2 * cm, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(1.2 * cm, H - 1.5 * cm, "ACA N.° 2 — Aprendizaje Automático aplicado a Ventas")
    c.setFont("Helvetica", 12)
    c.drawString(1.2 * cm, H - 2.5 * cm,
                 "CUN · Especialización en Inteligencia Artificial · Docente: Hamilton Rivera Flor")
    c.setFillColor(colors.black)

    y = H - 4.2 * cm
    col1_x, col2_x = 1.2 * cm, (W / 2) + 0.3 * cm
    col_ancho = (W / 2) - 1.6 * cm

    c.setFont("Helvetica-Bold", 13); c.drawString(col1_x, y, "Integrantes")
    y2 = _parrafo(c, integrantes, col1_x, y - 0.6 * cm, col_ancho)

    y2 -= 0.4 * cm
    c.setFont("Helvetica-Bold", 13); c.drawString(col1_x, y2, "Objetivo general")
    y2 = _parrafo(c, "Desarrollar, entrenar y desplegar redes neuronales profundas "
                      "(CNN, RNN secuencial y Transformer) que pronostiquen las "
                      "ventas diarias de una tienda minorista, integrando "
                      "preprocesamiento, entrenamiento, evaluacion comparativa y "
                      "una interfaz interactiva en Streamlit.", col1_x, y2 - 0.6 * cm, col_ancho)

    y2 -= 0.4 * cm
    c.setFont("Helvetica-Bold", 13); c.drawString(col1_x, y2, "Pregunta orientadora / Problema")
    y2 = _parrafo(c, "¿En qué medida distintas arquitecturas de redes neuronales "
                      "(convolucional, recurrente y basada en atención) logran "
                      "aprender patrones temporales en transacciones comerciales "
                      "para anticipar las ventas del día siguiente, y cuál ofrece "
                      "mejor desempeño?", col1_x, y2 - 0.6 * cm, col_ancho)

    y2 -= 0.4 * cm
    c.setFont("Helvetica-Bold", 13); c.drawString(col1_x, y2, "Dataset")
    y2 = _parrafo(c, "Sales Forecasting - Retail Superstore (train.csv): 9.800 "
                      "transacciones, periodo 2015-2018, agregadas a serie diaria "
                      "de ventas (variable 'Sales', en USD). Gestionado en 3 "
                      "bases SQLite: transacciones, analítica y predicciones.",
                  col1_x, y2 - 0.6 * cm, col_ancho)

    y2 -= 0.4 * cm
    c.setFont("Helvetica-Bold", 13); c.drawString(col1_x, y2, "Flujo metodológico")
    y2 = _parrafo(c, "1) EDA + gestión SQL  →  2) Ventanas deslizantes de 28 días "
                      "+ aumento de datos (ruido gaussiano)  →  3) Entrenamiento de "
                      "CNN, RNN (LSTM+GRU) y Transformer  →  4) Métricas comparativas "
                      "(MAE, RMSE, sMAPE, R2, F1)  →  5) Despliegue en Streamlit.",
                  col1_x, y2 - 0.6 * cm, col_ancho)

    # Columna derecha: arquitecturas + metricas
    y3 = y
    c.setFont("Helvetica-Bold", 13); c.drawString(col2_x, y3, "Arquitecturas comparadas")
    arquitecturas = (
        "CNN (Conv1D): 2 bloques convolucionales + pooling global; capta "
        "patrones locales de corto plazo en la ventana temporal.\n"
        "Secuencial (LSTM + GRU): capa LSTM seguida de GRU; modela "
        "dependencias temporales de mediano/largo plazo con memoria.\n"
        "Transformer: proyección + codificación posicional + auto-atención "
        "multi-cabeza (2 capas); pondera dinámicamente la importancia de "
        "cada día de la ventana."
    )
    y3 = _parrafo(c, arquitecturas, col2_x, y3 - 0.6 * cm, col_ancho)

    y3 -= 0.5 * cm
    c.setFont("Helvetica-Bold", 13); c.drawString(col2_x, y3, "Resultados principales (partición de prueba)")
    y3 -= 0.7 * cm
    c.setFont("Helvetica-Bold", 9)
    encabezados = ["Modelo", "MAE", "RMSE", "sMAPE%", "R2", "F1 alta venta"]
    anchos = [col_ancho * 0.34, col_ancho * 0.14, col_ancho * 0.14, col_ancho * 0.14, col_ancho * 0.1, col_ancho * 0.14]
    x = col2_x
    for h, a in zip(encabezados, anchos):
        c.drawString(x, y3, h); x += a
    y3 -= 0.4 * cm
    c.setFont("Helvetica", 9)
    for tipo, datos in info["resultados"].items():
        m = datos["metrics"]["test"]
        fila = [NOMBRES_MODELOS[tipo], f"{m['MAE']:.0f}", f"{m['RMSE']:.0f}",
                f"{m['sMAPE']:.1f}", f"{m['R2']:.2f}", f"{m['F1_alta_venta']:.2f}"]
        x = col2_x
        for val, a in zip(fila, anchos):
            c.drawString(x, y3, val); x += a
        y3 -= 0.38 * cm

    y3 -= 0.5 * cm
    c.setFont("Helvetica-Bold", 13); c.drawString(col2_x, y3, "Vista previa de la aplicación")
    y3 = _parrafo(c, "La app Streamlit permite: explorar los datos (EDA), "
                      "consultar las bases SQL, seleccionar la arquitectura "
                      "(CNN / Secuencial / Transformer), visualizar curvas de "
                      "entrenamiento y matriz de confusión, y generar el "
                      "pronóstico del día siguiente de forma interactiva.",
                  col2_x, y3 - 0.6 * cm, col_ancho)

    y3 -= 0.4 * cm
    c.setFont("Helvetica-Bold", 13); c.drawString(col2_x, y3, "Conclusiones")
    y3 = _parrafo(c, "Las tres arquitecturas superan ligeramente la base ingenua "
                      "semanal en términos de MAE, aunque el R2 negativo/cercano "
                      "a cero evidencia que la serie, al carecer de variables "
                      "exógenas (clima, promociones, inventario), limita la "
                      "capacidad predictiva de cualquier red; no debe emplearse "
                      "como sistema comercial sin ampliar las variables de entrada.",
                  col2_x, y3 - 0.6 * cm, col_ancho)

    # Pie de pagina
    c.setFillColor(colors.HexColor("#102A43"))
    c.rect(0, 0, W, 1.6 * cm, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica", 10)
    c.drawString(1.2 * cm, 0.6 * cm, app_url)
    c.drawRightString(W - 1.2 * cm, 0.6 * cm, video_url)

    c.showPage()
    c.save()
    return output
