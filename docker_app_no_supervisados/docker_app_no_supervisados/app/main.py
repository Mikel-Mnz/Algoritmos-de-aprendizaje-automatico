
"""
API de inferencia y monitoreo para modelos NO supervisados.

Expone dos modelos entrenados sobre el mismo esquema de features de clientes:
  - Segmentacion de clientes (KMeans)          -> /predict/segmento
  - Deteccion de anomalias (Isolation Forest)  -> /predict/anomalia

Ademas incluye endpoints de salud, metadatos y monitoreo de drift, siguiendo
la misma logica de contrato explicito, versionado y trazabilidad que se usa
para modelos supervisados (ver Tema 13), adaptada a la ausencia de etiqueta.
"""
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from joblib import load
from pydantic import BaseModel, Field

MODELS_DIR = Path(__file__).resolve().parent.parent / "modelos"

FEATURES = [
    "tx_count_30d",
    "avg_ticket",
    "night_ratio",
    "distinct_merchants_30d",
    "online_ratio",
    "avg_days_between_tx",
]

# ---------------------------------------------------------------------------
# Carga de artefactos (una sola vez, al iniciar el servicio)
# ---------------------------------------------------------------------------
seg_pipeline = load(MODELS_DIR / "segmentacion" / "v1" / "model.joblib")
seg_meta = json.loads((MODELS_DIR / "segmentacion" / "v1" / "metadata.json").read_text(encoding="utf-8"))

if_pipeline = load(MODELS_DIR / "anomalias" / "v1" / "model.joblib")
if_meta = json.loads((MODELS_DIR / "anomalias" / "v1" / "metadata.json").read_text(encoding="utf-8"))

# Referencia (ventana base / entrenamiento) usada para calcular drift en /monitor/drift
BASELINE_PATH = MODELS_DIR.parent / "ventana_base.csv"
baseline_df = pd.read_csv(BASELINE_PATH) if BASELINE_PATH.exists() else None

app = FastAPI(
    title="API de modelos no supervisados",
    version="1.0.0",
    description="Segmentacion (KMeans) y deteccion de anomalias (Isolation Forest) con monitoreo de drift.",
)


# ---------------------------------------------------------------------------
# Contrato de entrada / salida
# ---------------------------------------------------------------------------
class ClienteInput(BaseModel):
    tx_count_30d: float = Field(..., ge=0, description="Numero de transacciones en 30 dias")
    avg_ticket: float = Field(..., ge=0, description="Ticket promedio")
    night_ratio: float = Field(..., ge=0, le=1, description="Proporcion de transacciones nocturnas")
    distinct_merchants_30d: float = Field(..., ge=0, description="Comercios distintos en 30 dias")
    online_ratio: float = Field(..., ge=0, le=1, description="Proporcion de transacciones en linea")
    avg_days_between_tx: float = Field(..., ge=0, description="Dias promedio entre transacciones")


class BatchInput(BaseModel):
    registros: List[ClienteInput]


class MonitorWindowInput(BaseModel):
    """Cada lista debe tener el mismo orden/longitud que FEATURES por registro."""
    registros: List[ClienteInput]


def _to_frame(registros: List[ClienteInput]) -> pd.DataFrame:
    return pd.DataFrame([r.model_dump() for r in registros])[FEATURES]


# ---------------------------------------------------------------------------
# Endpoints de estado y metadatos
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/model-info")
def model_info():
    return {
        "segmentacion": {
            "model_name": seg_meta["model_name"],
            "model_version": seg_meta["model_version"],
            "algorithm": seg_meta["algorithm"],
            "n_clusters": seg_meta["n_clusters"],
            "metric_offline": seg_meta["metric_offline"],
            "metric_value": seg_meta["metric_value"],
        },
        "anomalias": {
            "model_name": if_meta["model_name"],
            "model_version": if_meta["model_version"],
            "algorithm": if_meta["algorithm"],
            "contamination": if_meta["contamination"],
            "metric_offline": if_meta["metric_offline"],
            "metric_value": if_meta["metric_value"],
        },
    }


# ---------------------------------------------------------------------------
# Inferencia: segmentacion (KMeans)
# ---------------------------------------------------------------------------
@app.post("/predict/segmento")
def predict_segmento(x: ClienteInput):
    X = pd.DataFrame([x.model_dump()])[FEATURES]
    cluster = int(seg_pipeline.predict(X)[0])

    scaler = seg_pipeline.named_steps["scaler"]
    kmeans = seg_pipeline.named_steps["kmeans"]
    X_scaled = scaler.transform(X)
    centro = kmeans.cluster_centers_[cluster]
    distancia = float(np.linalg.norm(X_scaled[0] - centro))

    return {
        "cluster": cluster,
        "distancia_al_centroide": round(distancia, 4),
        "n_clusters": seg_meta["n_clusters"],
        "model_version": seg_meta["model_version"],
    }


@app.post("/predict/segmento/batch")
def predict_segmento_batch(payload: BatchInput):
    if not payload.registros:
        raise HTTPException(status_code=422, detail="La lista 'registros' no puede estar vacia")
    X = _to_frame(payload.registros)
    clusters = seg_pipeline.predict(X).tolist()
    return {
        "clusters": clusters,
        "n": len(clusters),
        "distribucion": pd.Series(clusters).value_counts().sort_index().to_dict(),
        "model_version": seg_meta["model_version"],
    }


# ---------------------------------------------------------------------------
# Inferencia: deteccion de anomalias (Isolation Forest)
# ---------------------------------------------------------------------------
@app.post("/predict/anomalia")
def predict_anomalia(x: ClienteInput):
    X = pd.DataFrame([x.model_dump()])[FEATURES]
    scaler = if_pipeline.named_steps["scaler"]
    iforest = if_pipeline.named_steps["iforest"]
    X_scaled = scaler.transform(X)

    score = float(-iforest.score_samples(X_scaled)[0])  # mayor score = mas anomalo
    es_anomalia = bool(iforest.predict(X_scaled)[0] == -1)

    return {
        "anomaly_score": round(score, 4),
        "es_anomalia": es_anomalia,
        "score_referencia_media": round(if_meta["score_reference_mean"], 4),
        "score_referencia_std": round(if_meta["score_reference_std"], 4),
        "model_version": if_meta["model_version"],
    }


@app.post("/predict/anomalia/batch")
def predict_anomalia_batch(payload: BatchInput):
    if not payload.registros:
        raise HTTPException(status_code=422, detail="La lista 'registros' no puede estar vacia")
    X = _to_frame(payload.registros)
    scaler = if_pipeline.named_steps["scaler"]
    iforest = if_pipeline.named_steps["iforest"]
    X_scaled = scaler.transform(X)

    scores = (-iforest.score_samples(X_scaled)).tolist()
    flags = (iforest.predict(X_scaled) == -1).tolist()

    return {
        "n": len(scores),
        "tasa_anomalias": round(float(np.mean(flags)), 4),
        "anomaly_scores": [round(s, 4) for s in scores],
        "es_anomalia": flags,
        "model_version": if_meta["model_version"],
    }


# ---------------------------------------------------------------------------
# Monitoreo: comparacion de una ventana nueva contra la ventana base (PSI)
# ---------------------------------------------------------------------------
def _psi(base: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index sobre una variable numerica."""
    quantiles = np.linspace(0, 1, buckets + 1)
    edges = np.unique(np.quantile(base, quantiles))
    if len(edges) < 3:
        return 0.0
    base_counts, _ = np.histogram(base, bins=edges)
    actual_counts, _ = np.histogram(actual, bins=edges)

    base_pct = np.clip(base_counts / max(len(base), 1), 1e-6, None)
    actual_pct = np.clip(actual_counts / max(len(actual), 1), 1e-6, None)

    psi = np.sum((actual_pct - base_pct) * np.log(actual_pct / base_pct))
    return float(psi)


PSI_ALERTA = 0.25
PSI_OBSERVACION = 0.10


@app.post("/monitor/drift")
def monitor_drift(payload: MonitorWindowInput):
    if baseline_df is None:
        raise HTTPException(status_code=500, detail="No hay ventana base disponible en el servicio")
    if not payload.registros:
        raise HTTPException(status_code=422, detail="La lista 'registros' no puede estar vacia")

    actual = _to_frame(payload.registros)

    resultados = {}
    for col in FEATURES:
        psi_val = _psi(baseline_df[col].values, actual[col].values)
        if psi_val >= PSI_ALERTA:
            estado = "alerta"
        elif psi_val >= PSI_OBSERVACION:
            estado = "observacion"
        else:
            estado = "estable"
        resultados[col] = {"psi": round(psi_val, 4), "estado": estado}

    # Comparacion de comportamiento agregado del modelo (no solo de los datos)
    clusters_actual = seg_pipeline.predict(actual).tolist()
    scaler_if = if_pipeline.named_steps["scaler"]
    iforest = if_pipeline.named_steps["iforest"]
    tasa_anom_actual = float(np.mean(iforest.predict(scaler_if.transform(actual)) == -1))
    tasa_anom_base = if_meta["metric_value"]

    cambio_tasa_anomalias = round(tasa_anom_actual - tasa_anom_base, 4)
    alerta_modelo = abs(cambio_tasa_anomalias) >= 0.05

    return {
        "drift_por_variable": resultados,
        "distribucion_clusters_actual": pd.Series(clusters_actual).value_counts().sort_index().to_dict(),
        "tasa_anomalias_base": round(tasa_anom_base, 4),
        "tasa_anomalias_actual": round(tasa_anom_actual, 4),
        "cambio_tasa_anomalias": cambio_tasa_anomalias,
        "alerta_tasa_anomalias": alerta_modelo,
        "umbrales_psi": {"observacion": PSI_OBSERVACION, "alerta": PSI_ALERTA},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
