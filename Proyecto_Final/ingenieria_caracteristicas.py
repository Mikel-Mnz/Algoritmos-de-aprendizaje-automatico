"""Ingeniería de características del modelo de churn (Paso 5).

Está en un módulo para que el pipeline serializado (.joblib) pueda importarla
al cargarse desde el notebook o desde predictor.py.
"""
from sklearn.preprocessing import FunctionTransformer

COLUMNAS_SERVICIOS = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                      'TechSupport', 'StreamingTV', 'StreamingMovies']
NUEVAS_CARACTERISTICAS = ['Total_Servicios', 'Cliente_Nuevo']


def aplicar_ingenieria_caracteristicas(df):
    """Agrega Total_Servicios y Cliente_Nuevo a una copia del DataFrame."""
    df_temp = df.copy()

    # 1. Número total de servicios contratados
    df_temp['Total_Servicios'] = (df_temp[COLUMNAS_SERVICIOS] == 'Yes').sum(axis=1)

    # 2. Indicador de cliente nuevo (6 meses o menos)
    df_temp['Cliente_Nuevo'] = (df_temp['tenure'] <= 6).astype(int)

    return df_temp


def nombres_salida(transformador, columnas_entrada):
    """Nombres de salida del transformador: columnas de entrada más las nuevas."""
    return list(columnas_entrada) + NUEVAS_CARACTERISTICAS


def crear_transformador_ingenieria():
    """Primer paso del preprocesador: aplica la ingeniería dentro del pipeline."""
    return FunctionTransformer(aplicar_ingenieria_caracteristicas, feature_names_out=nombres_salida)
