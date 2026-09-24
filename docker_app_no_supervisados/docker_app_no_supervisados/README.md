# API de modelos no supervisados (segmentación + anomalías)

Contenido:
- `app/main.py` — servicio FastAPI (segmentación KMeans + detección de anomalías Isolation Forest + monitoreo de drift)
- `modelos/` — artefactos ya entrenados y serializados (joblib + metadata.json)
- `ventana_base.csv` — ventana de referencia usada por /monitor/drift
- `Dockerfile`, `docker-compose.yml`, `requirements.txt`
- `test_endpoints.sh` — batería de pruebas con curl

## Cómo ejecutar

```bash
docker compose up --build -d
curl http://127.0.0.1:8000/health
chmod +x test_endpoints.sh && ./test_endpoints.sh
```

Documentación interactiva: http://127.0.0.1:8000/docs

Detener:
```bash
docker compose down
```

Ver el notebook `Tema13_modelos_no_supervisados.ipynb` para la explicación teórica completa (matemática de K-Means e Isolation Forest, ciclo de vida MLOps adaptado a modelos no supervisados y monitoreo de drift) y el paso a paso que generó estos mismos artefactos.
