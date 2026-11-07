"""
ACA N.2 - Aprendizaje Automatico (CUN) | Pilar: Ventas
Pipeline unico que entrena y evalua TRES arquitecturas de redes neuronales
profundas sobre la misma serie de ventas diarias (Sales Forecasting - Retail
Superstore / train.csv):

    1) CNN   -> Conv1DForecast        (red neuronal convolucional 1D)
    2) RNN   -> SecuencialForecast    (red secuencial recurrente LSTM+GRU)
    3) TRF   -> TransformerForecast   (arquitectura basada en atencion)

El resto del flujo (lectura de datos, EDA, SQL, ventanas deslizantes,
normalizacion, aumento de datos, metricas y persistencia) se reutiliza de
forma identica para las tres arquitecturas, de modo que la comparacion sea
justa y reproducible.
"""
from __future__ import annotations
from pathlib import Path
import json
import random
import sqlite3

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

WINDOW = 28          # dias de contexto usados para predecir el dia siguiente
SEED = 42
MODELOS = ["cnn", "rnn", "transformer"]
NOMBRES_MODELOS = {
    "cnn": "CNN (Conv1D)",
    "rnn": "Secuencial (LSTM+GRU)",
    "transformer": "Transformer (atencion)",
}


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# PASO 1: Exploracion y gestion de datos (EDA + SQL)
# --------------------------------------------------------------------------
def read_sales(path: str = "train.csv") -> pd.DataFrame:
    candidatos = [Path(path), Path("datos") / path, Path("datos/train.csv")]
    archivo = next((p for p in candidatos if p.exists()), None)
    if archivo is None:
        raise FileNotFoundError(f"No se encontro el dataset en: {candidatos}")
    df = pd.read_csv(archivo, encoding="latin-1")
    for col in ("Order Date", "Ship Date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%d/%m/%Y", errors="coerce")
            if df[col].isna().mean() > 0.5:
                df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def daily_sales(df: pd.DataFrame) -> pd.DataFrame:
    serie = df.groupby("Order Date")["Sales"].sum().rename("sales")
    rango = pd.date_range(serie.index.min(), serie.index.max(), freq="D")
    daily = serie.reindex(rango, fill_value=0.0).rename_axis("date").reset_index()
    daily["weekday"] = daily["date"].dt.weekday
    daily["month"] = daily["date"].dt.month
    return daily


def sql_databases(df: pd.DataFrame, daily: pd.DataFrame, carpeta: str = "bases_sql") -> dict:
    """Crea tres bases SQLite independientes: transacciones, analitica y predicciones."""
    Path(carpeta).mkdir(exist_ok=True)
    rutas = {
        "transacciones": str(Path(carpeta) / "transacciones.sqlite"),
        "analitica": str(Path(carpeta) / "analitica.sqlite"),
        "predicciones": str(Path(carpeta) / "predicciones.sqlite"),
    }

    cols = ["Row ID", "Order ID", "Order Date", "Customer Name", "Segment",
            "Region", "Category", "Sub-Category", "Product Name", "Sales"]
    tx = df[[c for c in cols if c in df.columns]].copy()
    tx.columns = [c.lower().replace(" ", "_").replace("-", "_") for c in tx.columns]
    with sqlite3.connect(rutas["transacciones"]) as conn:
        tx.to_sql("ventas", conn, if_exists="replace", index=False)

    mensual = (daily.assign(periodo=daily["date"].dt.to_period("M").astype(str))
               .groupby("periodo")["sales"].sum().rename("ventas").reset_index())
    with sqlite3.connect(rutas["analitica"]) as conn:
        daily.rename(columns={"sales": "ventas"}).to_sql(
            "ventas_diarias", conn, if_exists="replace", index=False)
        mensual.to_sql("ventas_mensuales", conn, if_exists="replace", index=False)

    with sqlite3.connect(rutas["predicciones"]) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS predicciones (
            modelo TEXT, particion TEXT, fecha TEXT, real REAL, prediccion REAL)""")
        conn.commit()
    return rutas


def guardar_predicciones_sql(pred: pd.DataFrame, modelo: str, ruta_db: str) -> None:
    out = pred.copy()
    out["modelo"] = modelo
    with sqlite3.connect(ruta_db) as conn:
        out[["modelo", "particion", "fecha", "real", "prediccion"]].to_sql(
            "predicciones", conn, if_exists="append", index=False)


# --------------------------------------------------------------------------
# PASO 2: Preparacion, pipeline y aumento de datos (ingenieria de secuencias)
# --------------------------------------------------------------------------
def prepare(daily: pd.DataFrame, window: int = WINDOW) -> dict:
    n = len(daily)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    scaler = StandardScaler()
    scaler.fit(daily["sales"].values[:train_end].reshape(-1, 1))
    valores_norm = scaler.transform(daily["sales"].values.reshape(-1, 1)).flatten()

    X, y, fechas = [], [], []
    for i in range(window, n):
        X.append(valores_norm[i - window:i])
        y.append(valores_norm[i])
        fechas.append(daily["date"].iloc[i])
    X = np.array(X, dtype=np.float32)[..., None]   # (muestras, ventana, 1 variable)
    y = np.array(y, dtype=np.float32)
    fechas = pd.to_datetime(fechas)

    masks = {
        "train": fechas < daily["date"].iloc[train_end],
        "val": (fechas >= daily["date"].iloc[train_end]) & (fechas < daily["date"].iloc[val_end]),
        "test": fechas >= daily["date"].iloc[val_end],
    }
    limits = {
        "train_end": str(daily["date"].iloc[train_end - 1].date()),
        "val_end": str(daily["date"].iloc[val_end - 1].date()),
        "test_end": str(daily["date"].iloc[-1].date()),
    }
    return {"X": X, "y": y, "fechas": fechas, "masks": masks,
            "scaler": scaler, "limits": limits, "daily": daily, "window": window}


def _augment(Xb: torch.Tensor, sigma: float = 0.015) -> torch.Tensor:
    return Xb + torch.randn_like(Xb) * sigma


# --------------------------------------------------------------------------
# PASO 3: Arquitecturas de redes neuronales
# --------------------------------------------------------------------------
class Conv1DForecast(nn.Module):
    """Red Neuronal Convolucional (CNN) 1D: extrae patrones locales en la
    ventana temporal mediante convoluciones apiladas con pooling."""

    def __init__(self, in_ch: int = 1, canales=(16, 32), kernel: int = 5):
        super().__init__()
        capas, c_in = [], in_ch
        for c_out in canales:
            capas += [nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2),
                      nn.BatchNorm1d(c_out), nn.ReLU(), nn.Dropout(0.15)]
            c_in = c_out
        self.conv = nn.Sequential(*capas)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(c_in, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        # x: (batch, ventana, variables) -> Conv1d espera (batch, canales, ventana)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x).squeeze(-1)


class SecuencialForecast(nn.Module):
    """Red Neuronal Recurrente secuencial: LSTM seguida de GRU, la
    arquitectura originalmente planteada para el pilar de ventas."""

    def __init__(self, in_ch: int = 1, hidden: int = 32, dropout: float = 0.12):
        super().__init__()
        self.lstm = nn.LSTM(in_ch, hidden, num_layers=1, batch_first=True)
        self.gru = nn.GRU(hidden, hidden, num_layers=1, batch_first=True, dropout=0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        out, h = self.gru(out)
        out = self.drop(h[-1])
        return self.head(out).squeeze(-1)


# Alias retrocompatible: el notebook original importaba GRUForecast.
GRUForecast = SecuencialForecast


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerForecast(nn.Module):
    """Arquitectura basada en atencion (Transformer encoder): proyecta cada
    paso temporal a un espacio latente, agrega codificacion posicional y
    aplica auto-atencion multi-cabeza para pronosticar el dia siguiente."""

    def __init__(self, in_ch: int = 1, d_model: int = 32, nhead: int = 4,
                 num_layers: int = 2, dim_ff: int = 64, dropout: float = 0.1,
                 window: int = WINDOW):
        super().__init__()
        self.proyeccion = nn.Linear(in_ch, d_model)
        self.pos = PositionalEncoding(d_model, max_len=window + 1)
        capa = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(capa, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(d_model, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        z = self.proyeccion(x)
        z = self.pos(z)
        z = self.encoder(z)
        z = z.mean(dim=1)  # agregacion temporal (promedio de los estados atendidos)
        return self.head(z).squeeze(-1)


def construir_modelo(tipo: str, window: int = WINDOW) -> nn.Module:
    tipo = tipo.lower()
    if tipo == "cnn":
        return Conv1DForecast()
    if tipo in ("rnn", "secuencial"):
        return SecuencialForecast()
    if tipo in ("transformer", "trf", "atencion"):
        return TransformerForecast(window=window)
    raise ValueError(f"Tipo de modelo no reconocido: {tipo}")


# --------------------------------------------------------------------------
# Entrenamiento generico (valido para las 3 arquitecturas)
# --------------------------------------------------------------------------
def train_model(bundle: dict, tipo: str = "rnn", epochs: int = 35,
                 patience: int = 7, augment: bool = True, lr: float = 1e-3,
                 batch_size: int = 32, model: nn.Module | None = None):
    seed_all()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (model or construir_modelo(tipo, window=bundle["window"])).to(device)

    X, y, masks = bundle["X"], bundle["y"], bundle["masks"]
    Xtr = torch.tensor(X[masks["train"]])
    ytr = torch.tensor(y[masks["train"]])
    Xval = torch.tensor(X[masks["val"]]).to(device)
    yval = torch.tensor(y[masks["val"]]).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.SmoothL1Loss()
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)

    mejor_val, espera, mejor_estado = float("inf"), 0, None
    historial = []
    n = Xtr.shape[0]
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        perdida_ep = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xtr[idx].to(device), ytr[idx].to(device)
            if augment:
                xb = _augment(xb)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            perdida_ep += loss.item() * len(idx)
        perdida_ep /= n

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xval), yval).item()
        sched.step(val_loss)
        historial.append({"epoch": epoch, "train_loss": perdida_ep, "val_loss": val_loss})

        if val_loss < mejor_val - 1e-5:
            mejor_val, espera = val_loss, 0
            mejor_estado = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            espera += 1
            if espera >= patience:
                break

    if mejor_estado is not None:
        model.load_state_dict(mejor_estado)
    return model, pd.DataFrame(historial)


# --------------------------------------------------------------------------
# PASO 4: Evaluacion y metricas
# --------------------------------------------------------------------------
def _smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    denom = np.where(denom == 0, 1e-8, denom)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def evaluate(model: nn.Module, bundle: dict) -> tuple[dict, pd.DataFrame, float]:
    device = next(model.parameters()).device
    model.eval()
    scaler = bundle["scaler"]
    X, y, fechas, masks = bundle["X"], bundle["y"], bundle["fechas"], bundle["masks"]

    with torch.no_grad():
        pred_norm_all = model(torch.tensor(X).to(device)).cpu().numpy()

    real_all = scaler.inverse_transform(y.reshape(-1, 1)).flatten()
    pred_all = scaler.inverse_transform(pred_norm_all.reshape(-1, 1)).flatten()

    umbral = float(np.percentile(real_all[masks["train"]], 75))

    registros, metrics = [], {}
    for particion, mask in masks.items():
        r, p = real_all[mask], pred_all[mask]
        f = fechas[mask]
        base_sem = np.roll(r, 7)
        base_sem[:7] = r[:7]
        mae_base = mean_absolute_error(r, base_sem)

        r_clase = (r >= umbral).astype(int)
        p_clase = (p >= umbral).astype(int)
        acc = accuracy_score(r_clase, p_clase)
        f1 = f1_score(r_clase, p_clase, zero_division=0)
        cm = confusion_matrix(r_clase, p_clase, labels=[0, 1]).tolist()

        metrics[particion] = {
            "n": int(mask.sum()),
            "MAE": float(mean_absolute_error(r, p)),
            "MAE_baseline_semanal": float(mae_base),
            "RMSE": float(np.sqrt(mean_squared_error(r, p))),
            "sMAPE": _smape(r, p),
            "R2": float(r2_score(r, p)) if len(r) > 1 else float("nan"),
            "Accuracy_alta_venta": float(acc),
            "F1_alta_venta": float(f1),
            "confusion": cm,
        }
        for fe, re, pr in zip(f, r, p):
            registros.append({"particion": particion, "fecha": str(fe.date()), "real": float(re), "prediccion": float(pr)})

    pred_df = pd.DataFrame(registros)
    return metrics, pred_df, umbral


def forecast_next(model: nn.Module, bundle: dict) -> float:
    device = next(model.parameters()).device
    model.eval()
    ultima_ventana = bundle["X"][-1:]
    with torch.no_grad():
        pred_norm = model(torch.tensor(ultima_ventana).to(device)).cpu().numpy()
    return float(bundle["scaler"].inverse_transform(pred_norm.reshape(-1, 1))[0, 0])


# --------------------------------------------------------------------------
# PASO 5: Orquestacion completa (EDA -> SQL -> 3 modelos -> metricas -> persistencia)
# --------------------------------------------------------------------------
def run(epochs: int = 35, tipos=MODELOS, data_path: str = "train.csv") -> dict:
    seed_all()
    df = read_sales(data_path)
    daily = daily_sales(df)
    dbs = sql_databases(df, daily)
    bundle = prepare(daily, window=WINDOW)

    Path("pesos").mkdir(exist_ok=True)
    resultados = {}
    for tipo in tipos:
        model, historia = train_model(bundle, tipo=tipo, epochs=epochs)
        metrics, pred_df, umbral = evaluate(model, bundle)
        pronostico = forecast_next(model, bundle)

        torch.save(model.state_dict(), f"pesos/modelo_{tipo}.pt")
        historia.to_csv(f"pesos/historial_{tipo}.csv", index=False)
        pred_df.to_csv(f"pesos/predicciones_{tipo}.csv", index=False)
        guardar_predicciones_sql(pred_df, tipo, dbs["predicciones"])
        with open(f"pesos/metricas_{tipo}.json", "w", encoding="utf-8") as fh:
            json.dump({"metrics": metrics, "umbral": umbral, "pronostico_dia_siguiente": pronostico,
                       "n_parametros": sum(p.numel() for p in model.parameters() if p.requires_grad)},
                      fh, ensure_ascii=False, indent=2)

        resultados[tipo] = {
            "nombre": NOMBRES_MODELOS[tipo], "metrics": metrics, "historia": historia,
            "predicciones": pred_df, "umbral": umbral, "pronostico_dia_siguiente": pronostico,
            "n_parametros": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }

    scaler_params = {"mean": float(bundle["scaler"].mean_[0]), "scale": float(bundle["scaler"].scale_[0]),
                      "window": WINDOW}
    with open("pesos/normalizacion.json", "w", encoding="utf-8") as fh:
        json.dump(scaler_params, fh, indent=2)

    comparacion = pd.DataFrame({
        tipo: {
            "MAE_test": resultados[tipo]["metrics"]["test"]["MAE"],
            "RMSE_test": resultados[tipo]["metrics"]["test"]["RMSE"],
            "sMAPE_test": resultados[tipo]["metrics"]["test"]["sMAPE"],
            "R2_test": resultados[tipo]["metrics"]["test"]["R2"],
            "F1_alta_venta_test": resultados[tipo]["metrics"]["test"]["F1_alta_venta"],
            "parametros": resultados[tipo]["n_parametros"],
        } for tipo in tipos
    }).T
    comparacion.to_csv("pesos/comparacion_modelos.csv")

    return {"bundle": bundle, "dbs": dbs, "daily": daily, "df": df,
            "resultados": resultados, "comparacion": comparacion,
            "sql_bases": list(dbs.values())}


def cargar_modelo(tipo: str, window: int = WINDOW) -> nn.Module:
    model = construir_modelo(tipo, window=window)
    estado = torch.load(f"pesos/modelo_{tipo}.pt", map_location="cpu")
    model.load_state_dict(estado)
    model.eval()
    return model

if __name__ == "__main__":
    print("Iniciando el entrenamiento de los modelos...")
    run()
    print("Entrenamiento finalizado exitosamente. Carpeta 'pesos/' creada.")
