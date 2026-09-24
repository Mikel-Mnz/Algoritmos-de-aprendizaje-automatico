# Imagen base ligera con Python 3.11
FROM python:3.11-slim

# Evita archivos .pyc y fuerza salida de logs sin buffer (util para ver logs de uvicorn)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /srv

# 1) Dependencias primero -> aprovecha la cache de Docker si el codigo cambia pero no los requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Artefactos de modelo y codigo de la aplicacion
COPY modelos/ ./modelos/
COPY ventana_base.csv .
COPY app/ ./app/

EXPOSE 8000

# Chequeo de salud usado por docker-compose / orquestadores para saber si el servicio esta listo
HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
