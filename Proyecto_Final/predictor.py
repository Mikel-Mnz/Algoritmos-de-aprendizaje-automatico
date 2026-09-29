"""
PowerShell:

python Proyecto_Final/predictor.py

"""
import argparse
import json
import os
import pandas as pd
import joblib

# Obtener la ruta absoluta del directorio donde se ubica este script
DIR_ACTUAL = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO_DEFECTO = os.path.join(DIR_ACTUAL, 'pipeline_final_churn.joblib')

def aplicar_ingenieria_caracteristicas(df):
    """
    Replica la lógica de creación de variables derivada de la fase de entrenamiento.
    """
    df_temp = df.copy()
    columnas_servicios = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                          'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    for col in columnas_servicios:
        if col not in df_temp.columns:
            df_temp[col] = 'No'
            
    df_temp['Total_Servicios'] = (df_temp[columnas_servicios] == 'Yes').sum(axis=1)
    
    if 'tenure' in df_temp.columns:
        df_temp['Cliente_Nuevo'] = (pd.to_numeric(df_temp['tenure'], errors='coerce') <= 6).astype(int)
    else:
        df_temp['Cliente_Nuevo'] = 0
        
    return df_temp

def predecir_abandono(registro_json, ruta_modelo=RUTA_MODELO_DEFECTO):
    # 1. Cargar datos
    datos_dict = json.loads(registro_json)
    df_input = pd.DataFrame([datos_dict])
    
    # 2. Ingeniería de características
    df_procesado = aplicar_ingenieria_caracteristicas(df_input)
    
    # 3. Cargar pipeline
    if not os.path.exists(ruta_modelo):
        raise FileNotFoundError(f"No se encontró el archivo '{os.path.basename(ruta_modelo)}' en la ruta: {ruta_modelo}")
        
    pipeline = joblib.load(ruta_modelo)
    
    # 4. Inferencia
    probabilidad = pipeline.predict_proba(df_procesado)[0][1]
    
    # 5. Aplicar umbral operativo optimizado (0.30)
    umbral = 0.30
    clasificacion = "Abandono (Churn)" if probabilidad >= umbral else "Retención"
    
    # 6. Formatear salida requerida
    resultado = {
        "Probabilidad estimada de abandono": f"{probabilidad:.2%}",
        "Clasificación resultante": clasificacion,
        "Umbral utilizado": umbral,
        "Fecha / Versión del modelo": "v1.0 - Septiembre 2026",
        "Aviso": "El resultado mostrado es una estimación basada en un modelo estadístico y requiere revisión humana para la toma de decisiones."
    }
    
    return resultado

if __name__ == "__main__":
    json_defecto = json.dumps({
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", 
        "tenure": 3, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL", 
        "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No", 
        "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No", 
        "Contract": "Month-to-month", "PaperlessBilling": "Yes", 
        "PaymentMethod": "Electronic check", "MonthlyCharges": 53.85, "TotalCharges": 108.15
    })

    parser = argparse.ArgumentParser(description="Predictor de Abandono de Clientes (Churn)")
    parser.add_argument(
        "--datos", 
        type=str, 
        default=json_defecto,
        help="Datos del cliente en formato JSON string."
    )
    
    args = parser.parse_args()
    
    try:
        salida = predecir_abandono(args.datos)
        print("\n--- RESULTADO DE LA INFERENCIA ---")
        for clave, valor in salida.items():
            print(f"{clave}: {valor}")
        print("----------------------------------\n")
    except Exception as e:
        print(f"Error procesando la solicitud: {e}")