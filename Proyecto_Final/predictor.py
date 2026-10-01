"""Inferencia del modelo de churn (Pasos 24 a 26).

CÓMO USARLO
1. Instalar dependencias:  pip install -r requirements.txt
2. Generar el modelo:      ejecutar proyecto_final.ipynb (crea pipeline_final_churn.joblib y umbral_churn.json)
3. Predecir:               python predictor.py cliente_ejemplo.json

Entrada: archivo JSON con un cliente (objeto) o varios (lista de objetos) con las 19 columnas
de COLUMNAS_REQUERIDAS. En la misma carpeta deben estar ingenieria_caracteristicas.py,
pipeline_final_churn.joblib y umbral_churn.json.

Salida por cliente: probabilidad, clasificación, umbral, versión del modelo y aviso de revisión humana.
Los registros con datos inválidos no se predicen: se envían a revisión humana con el motivo.

Ejemplos de cliente_ejemplo.json:
  Registro 0: 2 meses, contrato mes a mes, fibra óptica      -> 75.04%, Abandono (Churn)
  Registro 1: 60 meses, contrato a dos años, DSL             -> 0.73%, No abandono
  Registro 2: método de pago no reconocido ("Digital wallet") -> revisión humana, sin predicción

Desde Python:
  from predictor import leer_entrada, predecir_nuevos_registros
  resultados, reporte = predecir_nuevos_registros(leer_entrada('cliente_ejemplo.json'))
"""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

CARPETA = Path(__file__).resolve().parent
RUTA_PIPELINE = CARPETA / 'pipeline_final_churn.joblib'
RUTA_CONFIG = CARPETA / 'umbral_churn.json'
AVISO = ('El resultado mostrado es una estimación basada en un modelo estadístico '
         'y requiere revisión humana para la toma de decisiones.')

COLUMNAS_REQUERIDAS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]
COLUMNAS_NUMERICAS = ['tenure', 'MonthlyCharges', 'TotalCharges']
SERVICIO_INTERNET = ['No', 'Yes', 'No internet service']
CATEGORIAS_VALIDAS = {
    'gender': ['Female', 'Male'],
    'SeniorCitizen': [0, 1],
    'Partner': ['Yes', 'No'],
    'Dependents': ['Yes', 'No'],
    'PhoneService': ['Yes', 'No'],
    'MultipleLines': ['No phone service', 'No', 'Yes'],
    'InternetService': ['DSL', 'Fiber optic', 'No'],
    'OnlineSecurity': SERVICIO_INTERNET,
    'OnlineBackup': SERVICIO_INTERNET,
    'DeviceProtection': SERVICIO_INTERNET,
    'TechSupport': SERVICIO_INTERNET,
    'StreamingTV': SERVICIO_INTERNET,
    'StreamingMovies': SERVICIO_INTERNET,
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'PaperlessBilling': ['Yes', 'No'],
    'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
}


def calcular_sha256(ruta):
    with open(ruta, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def cargar_artefactos():
    """Carga la configuración, verifica el hash SHA-256 del pipeline y solo entonces lo deserializa."""
    with open(RUTA_CONFIG, 'r', encoding='utf-8') as f:
        config = json.load(f)
    if calcular_sha256(RUTA_PIPELINE) != config['sha256'][RUTA_PIPELINE.name]:
        raise ValueError(f"El hash SHA-256 de '{RUTA_PIPELINE.name}' no coincide con el registrado; no se carga el archivo.")
    pipeline = joblib.load(RUTA_PIPELINE)
    return pipeline, config


def validar_datos_entrada(df):
    """
    Valida los datos de entrada recopilando anomalías sin detener la ejecución.
    Detecta: campos faltantes, tipos incorrectos, valores nulos, valores fuera de rango,
    categorías no reconocidas, columnas adicionales y estructura incompleta.
    'motivos_por_registro' lista los registros que deben ir a revisión humana en lugar de predecirse.
    Se permiten nulos en TotalCharges (clientes con tenure = 0) y en las categóricas: el pipeline los imputa.
    """
    reporte = {
        "valido": True,
        "campos_faltantes": [],
        "columnas_adicionales": [],
        "tipos_incorrectos": [],
        "valores_nulos": [],
        "valores_fuera_de_rango": [],
        "categorias_no_reconocidas": [],
        "motivos_por_registro": {}
    }
    motivos = {idx: [] for idx in df.index}

    def registrar(clave, mascara, mensaje):
        """Agrega el hallazgo al reporte general y a cada registro afectado."""
        mascara = np.asarray(mascara, dtype=bool)
        if mascara.any():
            reporte[clave].append(f"{mensaje}: {mascara.sum()} registros.")
            for idx in df.index[mascara]:
                motivos[idx].append(mensaje)

    # 1. Estructura incompleta y columnas adicionales
    faltantes = [col for col in COLUMNAS_REQUERIDAS if col not in df.columns]
    reporte["columnas_adicionales"] = [col for col in df.columns if col not in COLUMNAS_REQUERIDAS]
    if faltantes:
        reporte["campos_faltantes"] = faltantes
        for idx in df.index:
            motivos[idx].append(f"Faltan columnas requeridas: {faltantes}")

    # 2. Tipos incorrectos, nulos y valores fuera de rango en numéricas
    for col in [c for c in COLUMNAS_NUMERICAS if c in df.columns]:
        valores = df[col].replace(r'^\s*$', np.nan, regex=True)
        numerico = pd.to_numeric(valores, errors='coerce')
        registrar("tipos_incorrectos", numerico.isna() & valores.notna(), f"'{col}' no es numérico")
        if col != 'TotalCharges':
            registrar("valores_nulos", valores.isna(), f"'{col}' vacío")
    if 'tenure' in df.columns:
        tenure = pd.to_numeric(df['tenure'], errors='coerce')
        registrar("valores_fuera_de_rango", (tenure < 0) | (tenure > 72), "'tenure' fuera del rango [0, 72]")
    if 'MonthlyCharges' in df.columns:
        cargos = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
        registrar("valores_fuera_de_rango", cargos <= 0, "'MonthlyCharges' con valores <= 0")
    if 'TotalCharges' in df.columns:
        total = pd.to_numeric(df['TotalCharges'], errors='coerce')
        registrar("valores_fuera_de_rango", total < 0, "'TotalCharges' con valores negativos")

    # 3. Categorías no reconocidas en todas las variables categóricas y binarias
    for col, validas in CATEGORIAS_VALIDAS.items():
        if col in df.columns:
            invalidas = df[col].notna() & ~df[col].isin(validas)
            if invalidas.any():
                reporte["categorias_no_reconocidas"].append(
                    f"'{col}' tiene categorías no válidas: {list(df.loc[invalidas, col].unique())}")
                for idx in df.index[invalidas.to_numpy()]:
                    motivos[idx].append(f"'{col}' con categoría no reconocida")

    reporte["motivos_por_registro"] = {idx: m for idx, m in motivos.items() if m}
    reporte["valido"] = not reporte["motivos_por_registro"]
    return reporte


def predecir_nuevos_registros(df_nuevos):
    """Valida, predice los registros válidos y marca los inválidos para revisión humana."""
    pipeline, config = cargar_artefactos()
    umbral = config['umbral_operativo']
    reporte = validar_datos_entrada(df_nuevos)
    validos = ~df_nuevos.index.isin(list(reporte['motivos_por_registro']))

    resultados = pd.DataFrame(index=df_nuevos.index)
    resultados['Probabilidad_Churn'] = np.nan
    resultados['Clase_Predicha'] = pd.Series(pd.NA, index=df_nuevos.index, dtype='Int64')
    resultados['Decision'] = 'Revisión humana: datos inválidos'
    if validos.any():
        df_validos = df_nuevos.loc[validos, COLUMNAS_REQUERIDAS].copy()
        df_validos[COLUMNAS_NUMERICAS] = (df_validos[COLUMNAS_NUMERICAS]
                                          .replace(r'^\s*$', np.nan, regex=True)
                                          .apply(pd.to_numeric))
        y_proba = pipeline.predict_proba(df_validos)[:, 1]
        y_pred = (y_proba >= umbral).astype(int)
        resultados.loc[validos, 'Probabilidad_Churn'] = y_proba
        resultados.loc[validos, 'Clase_Predicha'] = y_pred
        resultados.loc[validos, 'Decision'] = np.where(y_pred == 1, 'Alerta de Abandono (Churn)', 'Cliente Retenido')
    resultados['Umbral'] = umbral
    resultados['Version_Modelo'] = f"{config['version_modelo']} ({config['fecha_entrenamiento']})"
    return resultados, reporte


def leer_entrada(ruta_json):
    """Lee un registro (objeto JSON) o varios (lista de objetos) como DataFrame."""
    with open(ruta_json, 'r', encoding='utf-8') as f:
        datos = json.load(f)
    return pd.DataFrame(datos if isinstance(datos, list) else [datos])


def imprimir_resultados(resultados, reporte):
    for idx, fila in resultados.iterrows():
        print("--- RESULTADO DE LA INFERENCIA ---")
        if idx in reporte['motivos_por_registro']:
            print(f"Registro {idx}: enviado a revisión humana sin predicción")
            print(f"Motivos: {'; '.join(reporte['motivos_por_registro'][idx])}")
        else:
            clasificacion = 'Abandono (Churn)' if fila['Clase_Predicha'] == 1 else 'No abandono'
            print(f"Probabilidad estimada de abandono: {fila['Probabilidad_Churn']:.2%}")
            print(f"Clasificación resultante: {clasificacion}")
        print(f"Umbral utilizado: {fila['Umbral']:.2f}")
        print(f"Fecha / Versión del modelo: {fila['Version_Modelo']}")
        print(f"Aviso: {AVISO}")
        print("----------------------------------")


def main():
    parser = argparse.ArgumentParser(description='Predicción de abandono de clientes (churn).')
    parser.add_argument('entrada', help='Archivo JSON con un registro o una lista de registros')
    args = parser.parse_args()
    resultados, reporte = predecir_nuevos_registros(leer_entrada(args.entrada))
    imprimir_resultados(resultados, reporte)


if __name__ == "__main__":
    main()
