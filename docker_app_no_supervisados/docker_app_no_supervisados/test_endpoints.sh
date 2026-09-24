#!/usr/bin/env bash
# Script de prueba end-to-end para la API de modelos no supervisados.
# Requiere que el contenedor este corriendo:  docker compose up -d --build
set -e
BASE_URL="http://127.0.0.1:8000"

echo "== /health =="
curl -s -X GET "$BASE_URL/health"; echo

echo "== /model-info =="
curl -s -X GET "$BASE_URL/model-info"; echo

echo "== /predict/segmento (cliente de uso medio) =="
curl -s -X POST "$BASE_URL/predict/segmento" \
  -H "Content-Type: application/json" \
  -d '{
    "tx_count_30d": 12,
    "avg_ticket": 110.5,
    "night_ratio": 0.18,
    "distinct_merchants_30d": 6,
    "online_ratio": 0.4,
    "avg_days_between_tx": 2.5
  }'; echo

echo "== /predict/anomalia (transaccion atipica) =="
curl -s -X POST "$BASE_URL/predict/anomalia" \
  -H "Content-Type: application/json" \
  -d '{
    "tx_count_30d": 70,
    "avg_ticket": 1800,
    "night_ratio": 0.85,
    "distinct_merchants_30d": 30,
    "online_ratio": 0.9,
    "avg_days_between_tx": 0.4
  }'; echo

echo "== /predict/segmento payload invalido (debe responder 422) =="
curl -s -o /dev/null -w "HTTP %{http_code}\n" -X POST "$BASE_URL/predict/segmento" \
  -H "Content-Type: application/json" \
  -d '{"tx_count_30d": 12, "night_ratio": 0.18}'

echo "Listo. Para ver la documentacion interactiva abre $BASE_URL/docs"
