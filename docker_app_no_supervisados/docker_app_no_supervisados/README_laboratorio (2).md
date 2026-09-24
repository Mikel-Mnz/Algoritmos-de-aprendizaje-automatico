# Manual de laboratorio
## Monitoreo y despliegue con FastAPI de modelos no supervisados

**Sesión técnica — audiencia universitaria**
**Duración estimada:** 2 horas (ajustable; ver desglose por fase)
**Modalidad:** laboratorio guiado con notebook + demo en vivo con Docker

---

## 0. Objetivos de la sesión

Al finalizar el laboratorio, el estudiante podrá:

1. Explicar por qué el ciclo de vida de un modelo **no supervisado** (sin etiqueta) exige criterios de validación y monitoreo distintos a los de un modelo supervisado.
2. Entrenar y evaluar un modelo de **segmentación (K-Means)** y uno de **detección de anomalías (Isolation Forest)** usando métricas internas (silhouette, Davies-Bouldin, tasa de contaminación).
3. Serializar un pipeline completo con `joblib` y documentarlo con metadatos trazables.
4. Exponer ambos modelos mediante una **API con FastAPI**, con contrato de entrada/salida explícito.
5. Implementar y ejecutar un **endpoint de monitoreo de drift** (PSI) y distinguir drift de datos de drift del comportamiento del modelo.
6. Levantar la API en un **contenedor Docker** y probarla con `curl` / Swagger UI.

---

## 1. Materiales entregados

| Archivo / carpeta | Contenido |
|---|---|
| `Tema13_modelos_no_supervisados.ipynb` | Notebook teórico-práctico completo, ya ejecutado, con matemática, código y resultados |
| `docker_app_no_supervisados.zip` | Aplicación FastAPI ya empaquetada, con los modelos entrenados, lista para `docker compose up` |
| `docker_app_no_supervisados/app/main.py` | Código fuente de la API (dentro del zip) |
| `docker_app_no_supervisados/modelos/` | Artefactos serializados (`model.joblib` + `metadata.json`) de segmentación y anomalías |
| `docker_app_no_supervisados/test_endpoints.sh` | Script de pruebas automatizadas con `curl` |

> **Antes de la sesión**, descomprime `docker_app_no_supervisados.zip` en una carpeta de trabajo. El notebook puede ejecutarse aparte; no depende de que el zip esté descomprimido.

---

## 2. Prerrequisitos técnicos (verificar antes de iniciar)

| Requisito | Verificación | Instalación si falta |
|---|---|---|
| Python 3.10+ | `python3 --version` | https://www.python.org/downloads/ |
| Jupyter / JupyterLab | `jupyter --version` | `pip install jupyterlab` |
| Docker Desktop / Docker Engine + Compose v2 | `docker --version` y `docker compose version` | https://docs.docker.com/get-docker/ |
| Editor de texto o IDE (VS Code recomendado) | — | https://code.visualstudio.com/ |
| Conexión a internet (solo para instalar dependencias la primera vez) | — | — |

**Recomendación logística:** pedir a los estudiantes que verifiquen `docker --version` y `docker compose version` **antes** de llegar al laboratorio; es la causa más común de retraso en sesiones con Docker.

---

## 2.1 Configuración alternativa para entornos con Anaconda

Varios laboratorios universitarios usan **Anaconda / Miniconda** como gestor de entornos por defecto. Esta sección reemplaza los pasos de instalación con `pip` de las fases 2 y 3 cuando el equipo del estudiante tiene Anaconda instalado. El resto del manual (notebook, API, Docker) no cambia.

### Verificación previa

```bash
conda --version
conda info --envs
```

### Paso 1 — Crear un entorno dedicado para el laboratorio

Se recomienda **no** usar el entorno `base` para evitar conflictos de versiones con otros cursos o proyectos.

```bash
conda create -n lab-no-supervisado -c conda-forge python=3.11 -y
conda activate lab-no-supervisado
```

> **Importante:** se especifica `-c conda-forge` de forma explícita. El canal `defaults` de Anaconda no siempre tiene indexada la versión de Python solicitada (o puede estar desactualizado/restringido en instalaciones institucionales), lo que produce errores de resolución confusos como el descrito en la tabla de troubleshooting (`PackagesNotFoundInChannelsError`). Usar `conda-forge` evita ese problema en la gran mayoría de los casos.

### Paso 2 — Instalar las dependencias

Las librerías del notebook y de la API se instalan de forma mixta: las más pesadas (`scikit-learn`, `pandas`, `numpy`) desde el canal `conda-forge`, y las específicas del servicio web (`fastapi`, `uvicorn`, `httpx`) con `pip`, ya que no siempre tienen paquete conda actualizado.

```bash
conda install -n lab-no-supervisado -c conda-forge scikit-learn pandas numpy matplotlib jupyterlab ipykernel -y
conda activate lab-no-supervisado
pip install fastapi uvicorn[standard] pydantic joblib httpx
```

> **Nota para el instructor:** si el laboratorio cuenta con conexión a internet limitada, puede prepararse de antemano un archivo `environment.yml` (ver más abajo) y distribuirlo a los estudiantes junto con el resto del material, de modo que la creación del entorno sea un solo comando.

**`environment.yml` sugerido** (opcional, para distribuir antes de la sesión):

```yaml
name: lab-no-supervisado
channels:
  - conda-forge
dependencies:
  - python=3.11
  - scikit-learn
  - pandas
  - numpy
  - matplotlib
  - jupyterlab
  - ipykernel
  - pip
  - pip:
      - fastapi
      - "uvicorn[standard]"
      - pydantic
      - joblib
      - httpx
```

Con ese archivo, el estudiante solo necesita ejecutar:

```bash
conda init powershell

conda env create -f environment.yml
conda activate lab-no-supervisado
```

### Alternativa si `conda` sigue fallando durante la sesión

Si algún equipo tiene una instalación de Anaconda con el canal `defaults` bloqueado o desactualizado y no logra resolver el entorno ni con `conda-forge`, la vía más rápida para no perder tiempo de laboratorio es usar `venv` (incluido en Python estándar) en lugar de conda, solo para esta sesión:

```bash
python3 -m venv lab-no-supervisado
# En Linux/macOS:
source lab-no-supervisado/bin/activate
# En Windows (PowerShell):
lab-no-supervisado\Scripts\Activate.ps1

pip install scikit-learn pandas numpy matplotlib jupyterlab ipykernel fastapi "uvicorn[standard]" pydantic joblib httpx
python -m ipykernel install --user --name lab-no-supervisado --display-name "Python (lab-no-supervisado)"
```

Esto no requiere resolver canales de conda en absoluto y usa directamente el Python del sistema (siempre que sea 3.10+).

### Paso 3 — Registrar el entorno como kernel de Jupyter

Este paso es el que con más frecuencia se olvida en laboratorios con Anaconda: si no se registra el kernel, JupyterLab puede seguir ejecutando el notebook con el entorno `base` en lugar del entorno recién creado.

```bash
python -m ipykernel install --user --name lab-no-supervisado --display-name "Python (lab-no-supervisado)"
```

Al abrir `Tema13_modelos_no_supervisados.ipynb`, verificar en el menú **Kernel → Change Kernel** que esté seleccionado **"Python (lab-no-supervisado)"** antes de ejecutar cualquier celda.

### Paso 4 — Ejecutar JupyterLab desde el entorno conda

```bash
conda activate lab-no-supervisado
jupyter lab
```

### Sobre Docker y Anaconda

El contenedor Docker de la Fase 5 **no depende de Anaconda**: la imagen se construye con `python:3.11-slim` e instala sus dependencias con `pip` dentro del propio contenedor, de forma aislada al entorno conda del anfitrión. Esto significa que:

- No es necesario tener el entorno conda activado para ejecutar `docker compose up --build -d`.
- Los estudiantes que trabajen únicamente con Anaconda (sin usar Docker) pueden completar las Fases 1 a 4 por completo dentro de su entorno conda, y solo necesitarán Docker para la Fase 5.
- Si se desea evitar Docker por completo en un laboratorio con recursos limitados, la API puede levantarse directamente desde el entorno conda con:
  ```bash
  cd docker_app_no_supervisados
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
  ```
  En ese caso, el script `test_endpoints.sh` funciona igual, ya que solo depende de que el servicio responda en `http://127.0.0.1:8000`.

---

## 2.2 Si tu equipo tiene Python 3.14

Python 3.14 es más reciente que la versión usada dentro de la imagen Docker (`python:3.11-slim`), y esto tiene dos implicaciones distintas según en qué parte del laboratorio te encuentres:

### La Fase 5 (Docker) no se ve afectada

El contenedor construye **su propio entorno Python 3.11 desde cero**, aislado del sistema operativo anfitrión. Si tu máquina tiene Python 3.14 instalado, `docker compose up --build` seguirá funcionando exactamente igual, porque el `Dockerfile` nunca usa el Python de tu equipo — usa el que viene dentro de la imagen `python:3.11-slim`. **No es necesario hacer ningún cambio para la Fase 5.**

### Las Fases 1 a 4 (notebook local) sí requieren un ajuste

Si vas a ejecutar `Tema13_modelos_no_supervisados.ipynb` directamente con Python 3.14 (fuera de Docker), ten en cuenta que versiones fijas y antiguas de algunas librerías (por ejemplo `scikit-learn==1.5.2`, `pydantic==2.9.2`, si se instalaran con esos pines exactos) **no publican instaladores (*wheels*) para Python 3.14** y la instalación fallaría. Hay dos opciones, de la más recomendada a la más rápida:

**Opción A — recomendada: usar Python 3.11 en un entorno aislado, igual que la imagen Docker.**
Así el comportamiento del notebook y el de la API dockerizada quedan alineados, evitando diferencias sutiles de versión entre ambos. Con conda:
```bash
conda create -n lab-no-supervisado -c conda-forge python=3.11 -y
conda activate lab-no-supervisado
```
Esto es independiente de que el sistema operativo tenga Python 3.14 instalado como versión por defecto — conda instala su propio intérprete 3.11 dentro del entorno, sin afectar el resto del sistema.

**Opción B — usar Python 3.14 tal cual, con versiones de librerías actualizadas.**
Si prefieres no instalar Python 3.11 en paralelo, puedes ejecutar el notebook con 3.14 instalando versiones recientes (sin pin exacto), que sí publican wheels para 3.14:
```bash
python3.14 -m venv lab-no-supervisado
source lab-no-supervisado/bin/activate   # En Windows: lab-no-supervisado\Scripts\Activate.ps1

pip install "scikit-learn>=1.9" "pandas>=2.2" "numpy>=1.26" matplotlib jupyterlab ipykernel \
            "fastapi>=0.119" "uvicorn[standard]>=0.32" "pydantic>=2.12" joblib httpx
python -m ipykernel install --user --name lab-no-supervisado --display-name "Python 3.14 (lab-no-supervisado)"
```
> Con esta opción, algunos resultados numéricos del notebook (silhouette, scores de anomalía) pueden diferir en la última cifra decimal respecto a los mostrados en el notebook de referencia, por el cambio de versión de `scikit-learn`; la interpretación y las conclusiones no cambian.

**Recomendación para el instructor de laboratorio:** si varios estudiantes tienen Python 3.14 preinstalado (versión por defecto en instalaciones recientes de Anaconda/Miniconda), conviene anunciar la Opción A desde la convocatoria de la sesión, para que todo el grupo trabaje con la misma versión de Python y se eviten diferencias de versión en vivo durante el laboratorio.

---

## 3. Estructura de la sesión (guion para el instructor)

| Fase | Duración aprox. | Actividad |
|---|---|---|
| Fase 1 | 10 min | Encuadre teórico: diferencias supervisado vs. no supervisado |
| Fase 2 | 30 min | Recorrido guiado del notebook (secciones 1 a 6: matemática + entrenamiento) |
| Fase 3 | 20 min | Serialización y construcción de la API (secciones 7 a 8 del notebook) |
| Fase 4 | 25 min | Monitoreo de drift: ejecutar y modificar el endpoint `/monitor/drift` |
| Fase 5 | 25 min | Dockerización: construir, levantar y probar la API en contenedor |
| Fase 6 | 10 min | Ejercicio propuesto + cierre y preguntas de reflexión |

---

## 4. Paso a paso

### Fase 1 — Encuadre teórico (10 min)

1. Proyectar la tabla comparativa de la sección 1 del notebook (*supervisado vs. no supervisado*).
2. Plantear la pregunta disparadora a la audiencia: **"¿Cómo sabemos que un modelo de segmentación 'funciona bien' si no existe una etiqueta correcta contra la cual compararlo?"**
3. Recoger 2-3 respuestas antes de continuar; conectar las respuestas con las métricas internas que se verán en la Fase 2 (silhouette, Davies-Bouldin, tasa de contaminación).

### Fase 2 — Recorrido del notebook: matemática y entrenamiento (30 min)

1. Abrir `Tema13_modelos_no_supervisados.ipynb` en Jupyter:
   ```bash
   jupyter lab Tema13_modelos_no_supervisados.ipynb
   ```
2. Recorrer en orden las secciones **1 a 4** (introducción, fundamentos matemáticos, ciclo de vida, contexto de los datos), deteniéndose en:
   - La función objetivo de K-Means y la fórmula de silhouette (sección 2.1).
   - La intuición de Isolation Forest: "un punto anómalo se aísla en menos particiones" (sección 2.2), antes de mostrar la fórmula formal.
3. Ejecutar en vivo las celdas de la **sección 4** (generación del dataset sintético) y comentar por qué se usa un dataset sintético con estructura realista en lugar de uno descargado (nota metodológica incluida en el notebook).
4. Ejecutar las celdas de las **secciones 5 y 6** (entrenamiento de K-Means e Isolation Forest). Pausar en el gráfico de método del codo / silhouette y discutir con la audiencia:
   > *"El óptimo estadístico da k=2, pero elegimos k=3. ¿Por qué es una decisión legítima y no un error?"*
5. **Punto de verificación (checkpoint 1):** pedir a los estudiantes que ejecuten hasta la sección 6 y compartan en pantalla su gráfico de silhouette — confirma que el entorno está funcionando antes de avanzar.

### Fase 3 — Serialización y construcción de la API (20 min)

1. Ejecutar la **sección 7** del notebook (serialización con `joblib` + `metadata.json`) y mostrar el contenido de un `metadata.json` generado:
   ```bash
   cat modelos/segmentacion/v1/metadata.json
   ```
2. Resaltar tres elementos que **siempre** deben documentarse en el metadata: (a) la métrica offline y su valor, (b) la decisión de negocio sobre parámetros clave (aquí, el valor de *k*), (c) las versiones de librerías usadas.
3. Ejecutar la **sección 8** del notebook (código de `app/main.py` y pruebas con `TestClient`). Explicar el contrato de cada endpoint antes de correrlo:
   - `/predict/segmento` → cluster + distancia al centroide
   - `/predict/anomalia` → score + bandera + score de referencia
   - Ambos con variante `/batch`
4. Mostrar en vivo la respuesta `422` ante un payload incompleto — punto pedagógico clave: **el contrato explícito rechaza datos ambiguos en lugar de inventar un valor**.

### Fase 4 — Monitoreo de drift (25 min)

1. Ejecutar la **sección 9** del notebook y explicar la fórmula de PSI en el pizarrón/diapositiva antes de correr el código.
2. Ejecutar la celda de la "ventana similar" y luego la de "ventana con drift inducido"; comparar ambos resultados JSON en pantalla.
3. **Ejercicio guiado (en vivo, con participación de la audiencia):** modificar la celda de la ventana con drift para alterar una variable distinta (por ejemplo, `distinct_merchants_30d` en lugar de `avg_ticket`) y pedir a un estudiante que prediga, antes de ejecutar, qué variable mostrará alerta.
4. Cerrar la fase relacionando el resultado con la tabla de niveles de respuesta (aviso / alerta / incidente) de la sección 9.4.

### Fase 5 — Dockerización y prueba de endpoints (25 min)

1. Desde una terminal, ubicarse en la carpeta descomprimida:
   ```bash
   cd docker_app_no_supervisados
   ```
2. Construir y levantar el contenedor:
   ```bash
   docker compose up --build -d
   ```
3. Verificar que el contenedor esté saludable:
   ```bash
   docker ps
   curl http://127.0.0.1:8000/health
   ```
4. Ejecutar la batería de pruebas:
   ```bash
   chmod +x test_endpoints.sh
   ./test_endpoints.sh
   ```
5. Abrir la documentación interactiva generada automáticamente por FastAPI y probar un endpoint manualmente desde el navegador:
   ```
   http://127.0.0.1:8000/docs
   ```
6. **Punto de verificación (checkpoint 2):** cada estudiante (o equipo) debe mostrar en pantalla la respuesta de `/predict/anomalia` para un payload que ellos mismos construyan.
7. Al finalizar, detener el contenedor:
   ```bash
   docker compose down
   ```

### Fase 6 — Ejercicio propuesto y cierre (10 min)

**Ejercicio para entregar (individual o en parejas, fuera de la sesión o como cierre):**

1. Modificar `app/main.py` para agregar un nuevo endpoint `/predict/segmento/explicacion` que, además del cluster, devuelva la distancia del cliente a **los tres centroides** (no solo al asignado).
2. Reconstruir la imagen de Docker (`docker compose up --build -d`) y probar el nuevo endpoint con `curl`.
3. Responder por escrito (máximo media página): *¿en qué escenario de negocio sería útil conocer la distancia a los tres centroides y no solo al asignado?*

**Preguntas de cierre para discusión grupal** (tomadas de la sección 12 del notebook):

- ¿Qué riesgo se introduce si se despliega un modelo de segmentación sin documentar por qué se eligió *k* frente a la alternativa que maximiza silhouette?
- ¿Por qué el drift de la tasa de anomalías puede ser una señal más útil que el drift de una sola variable de entrada?
- Si el servicio de anomalías responde siempre con latencia baja, ¿por qué eso no garantiza que sus alertas sigan siendo confiables seis meses después?

---

## 5. Solución de problemas comunes (troubleshooting)

| Síntoma | Causa probable | Solución |
|---|---|---|
| `docker compose` no reconocido | Versión antigua de Docker (compose v1) | Usar `docker-compose` (con guion) o actualizar Docker Desktop |
| El contenedor levanta pero `/health` no responde | El build aún no termina (`start_period` del healthcheck) | Esperar 10-15 segundos y reintentar; revisar `docker logs modelos_no_supervisados_api` |
| Error `ModuleNotFoundError` al ejecutar el notebook | Faltan dependencias en el entorno local | `pip install fastapi uvicorn scikit-learn pandas numpy joblib httpx` |
| `curl: (7) Failed to connect` | El contenedor no quedó expuesto en el puerto 8000 | Verificar con `docker ps` que el mapeo `8000:8000` esté activo |
| Los resultados de PSI cambian mucho entre ejecuciones con muestras pequeñas | Tamaño de muestra insuficiente para estimar los *buckets* de PSI de forma estable | Usar ventanas de al menos 300-400 registros en las pruebas (ver nota en la sección 9.3 del notebook) |
| Puerto 8000 ocupado por otro proceso | Otro servicio local usa el mismo puerto | Cambiar el mapeo en `docker-compose.yml` a, por ejemplo, `"8001:8000"` |
| El notebook se ejecuta pero usa versiones de librerías distintas a las esperadas | JupyterLab está usando el kernel del entorno `base` en lugar del entorno conda creado para el laboratorio | Verificar **Kernel → Change Kernel** y seleccionar el kernel registrado en el Paso 3 de la sección 2.1 |
| `conda install` muy lento o se queda "resolviendo entorno" | El solver clásico de conda puede ser lento con muchos paquetes | Usar `conda config --set solver libmamba` (requiere `conda install -n base conda-libmamba-solver`) o instalar con `mamba` si está disponible |
| Conflicto de versiones entre `pip` y `conda` dentro del mismo entorno | Se instalaron paquetes con `pip` antes que con `conda`, o en orden mezclado | Seguir el orden del Paso 2: primero `conda install`, después `pip install`; si persiste, recrear el entorno desde `environment.yml` |
| `PackagesNotFoundInChannelsError: ... The following packages are not available from current channels: - 3.14` (o cualquier número de versión aparece como si fuera un "paquete") al crear el entorno | El canal `defaults` no tiene indexada esa versión de Python — puede estar desactualizado, o el acceso al canal `defaults` está restringido/limitado en la instalación de Anaconda de la institución (cambio de licenciamiento de canales de Anaconda que afecta a organizaciones) | 1) Volver a crear el entorno indicando el canal explícitamente: `conda create -n lab-no-supervisado -c conda-forge python=3.11 -y`. <br>2) Si el error persiste, listar las versiones realmente disponibles con `conda search -c conda-forge python` y usar una de esa lista (por ejemplo `python=3.11` o `python=3.12`). <br>3) Como alternativa rápida sin depender de conda para resolver Python, usar `python -m venv` en vez de conda (ver nota siguiente) |
| `pip install` falla con errores de compilación o "no matching distribution found" para `scikit-learn`, `pydantic` u otra librería, en un equipo con **Python 3.14** | Las versiones fijadas en `requirements.txt` son anteriores a que esas librerías publicaran wheels para 3.14 | Seguir la sección 2.2 ("Si tu equipo tiene Python 3.14"): usar Python 3.11 en el entorno (Opción A, recomendada) o instalar versiones actualizadas sin pin exacto (Opción B) |

---

## 6. Cierre de la sesión — checklist para el instructor

- [ ] Todos los equipos ejecutaron el notebook hasta la sección de monitoreo sin errores.
- [ ] Todos los equipos lograron levantar el contenedor y obtener una respuesta válida de `/predict/anomalia`.
- [ ] Se discutieron en grupo al menos dos de las preguntas de cierre.
- [ ] Se asignó el ejercicio de la Fase 6 con fecha de entrega.
- [ ] Se recordó apagar los contenedores (`docker compose down`) antes de cerrar sesión.

---

*Material de apoyo: `Tema13_modelos_no_supervisados.ipynb` (teoría y práctica completa) y `docker_app_no_supervisados.zip` (aplicación lista para desplegar).*
