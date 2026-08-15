# Algoritmos de Aprendizaje Automático

Repositorio colaborativo destinado al desarrollo y almacenamiento de las actividades y prácticas de la materia de **Algoritmos de aprendizaje automático**.

## Estructura del Proyecto

El proyecto sigue una estructura de tipo *monorepo*, pensado para mantener todas las actividades centralizadas. En la raíz del proyecto se encuentran las configuraciones globales, y cada actividad tiene su propio directorio aislado para sus Jupyter Notebooks y datasets (`.csv`).

```text
📦 Algoritmos-de-aprendizaje-automatico
 ┣ 📂 Actividad_01/
 ┃ ┣ 📜 notebook.ipynb
 ┃ ┗ 📜 dataset.csv
 ┣ 📂 Actividad_02/
 ┃ ┣ 📜 notebook.ipynb
 ┃ ┗ 📜 dataset.csv
 ┣ 📜 .gitignore
 ┣ 📜 README.md
 ┗ 📜 requirements.txt
```

## Configuración Inicial

Para garantizar que todos trabajemos con las mismas versiones de las librerías y evitar conflictos, es necesario utilizar un entorno virtual. 

Sigue estos pasos para configurar tu entorno local:

### 1. Clonar el repositorio
Abre tu terminal y clona este repositorio en tu máquina local:
```bash
git clone https://github.com/Mikel-Mnz/Algoritmos-de-aprendizaje-automatico.git
cd Algoritmos-de-aprendizaje-automatico
```

### 2. Crear un entorno virtual (.venv)
Crea un entorno virtual dentro de la carpeta raíz del proyecto. 

- **En Windows:**
  ```bash
  python -m venv .venv
  ```
- **En macOS y Linux:**
  ```bash
  python3 -m venv .venv
  ```

### 3. Activar el entorno virtual
Una vez creado, debes activarlo. **Asegúrate de hacer esto cada vez que vayas a trabajar en el proyecto.**

- **En Windows:**
  ```bash
  .venv\Scripts\activate
  ```
- **En macOS y Linux:**
  ```bash
  source .venv/bin/activate
  ```

### 4. Instalar las dependencias
Con el entorno activado, instala todas las librerías necesarias ejecutando:
```bash
pip install -r requirements.txt
```

## Colaboración y Buenas Prácticas

- **Rutas de archivos:** Al cargar archivos `.csv` en tus libretas de Jupyter, utiliza siempre rutas relativas (ej. `pd.read_csv('dataset.csv')`). Esto asegura que el código funcione en la computadora de cualquier miembro del equipo.
- **Actualización de dependencias:** Si necesitas instalar una nueva librería, instálala usando `pip install <paquete>` y luego actualiza el archivo `requirements.txt` ejecutando:
  ```bash
  pip freeze > requirements.txt
  ```