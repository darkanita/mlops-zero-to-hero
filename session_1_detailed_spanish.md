# Sesión 1: Fundamentos MLOps y Configuración del Entorno - Plan de Enseñanza Detallado

**🌍 Idiomas:** [English](session-1-detailed.md) | [Español](session-1-detailed-ES.md)

**Duración:** 4 horas  
**Formato:** Teoría (1.5h) + Laboratorio Práctico (2.5h)  
**Tamaño de Clase:** 8-20 estudiantes  
**Prerrequisitos:** Python básico, conceptos ML, fundamentos Git

---

## 🎯 Objetivos de Aprendizaje de la Sesión

Al finalizar esta sesión, los estudiantes serán capaces de:
- Explicar qué es MLOps y por qué es crítico para proyectos ML modernos
- Identificar los desafíos clave en la gestión del ciclo de vida ML
- Configurar un entorno de desarrollo MLOps profesional usando herramientas modernas
- Crear su primer experimento MLFlow con seguimiento apropiado
- Aplicar mejores prácticas para organización de proyectos ML

---

## 📚 Preparación Pre-Sesión

### Configuración del Instructor
- [ ] Entorno demo MLFlow ejecutándose
- [ ] Datasets de muestra descargados
- [ ] Repositorio de código preparado
- [ ] Conexión de internet de respaldo (para instalación uv)
- [ ] Slides de presentación listos

### Prerrequisitos del Estudiante
- [ ] Laptop con privilegios de administrador
- [ ] Git instalado y configurado
- [ ] Cuenta GitHub creada
- [ ] Familiaridad básica con terminal/línea de comandos
- [ ] Python 3.8+ instalado (cualquier versión por ahora)

---

## 🕐 Parte 1: Introducción a MLOps (30 minutos)

### Gancho de Apertura (5 minutos)
**Comenzar con un escenario real:**
*"Imaginen que son científicos de datos en Spotify. Han construido un modelo de recomendación increíble que funciona perfectamente en su laptop con 95% de precisión. Pero cuando intentan desplegarlo a producción para servir a 400 millones de usuarios, todo se rompe. Las predicciones del modelo son diferentes, el código falla, y nadie puede reproducir sus resultados. Por esto necesitamos MLOps."*

### Contenido Central (20 minutos)

#### ¿Qué es MLOps? (8 minutos)
**Definición:**
> MLOps es la práctica de combinar el desarrollo de Machine Learning (ML) con principios DevOps para automatizar y mejorar la entrega continua de modelos ML a producción.

**Componentes Clave:**
1. **Desarrollo de Modelos** - Seguimiento de experimentos, control de versiones
2. **Despliegue de Modelos** - Pipelines automatizados, containerización
3. **Monitoreo de Modelos** - Seguimiento de rendimiento, detección de deriva
4. **Gobernanza de Modelos** - Cumplimiento, seguridad, aprobaciones

**Analogía Visual:**
- Software Tradicional: Código → Construir → Probar → Desplegar → Monitorear
- MLOps: Datos + Código + Modelo → Experimentar → Validar → Desplegar → Monitorear → Reentrenar

#### Por qué MLOps es Importante (7 minutos)
**Estadísticas de la Crisis ML:**
- 87% de proyectos ML nunca llegan a producción
- Tiempo promedio de despliegue: 6-12 meses
- 90% de modelos se degradan en 6 meses sin monitoreo

**Problemas Reales que MLOps Resuelve:**
1. **"Funciona en mi máquina"** → Entornos reproducibles
2. **"¿Dónde está la versión 2.3 del modelo?"** → Versionado de modelos
3. **"¿El modelo sigue siendo preciso?"** → Monitoreo automatizado
4. **"¿Quién aprobó este modelo?"** → Flujos de gobernanza

**Impacto Empresarial:**
- Netflix: MLOps reduce el tiempo de despliegue de modelos de meses a días
- Uber: Reentrenamiento automatizado previene caída del 40% en precisión del modelo
- Airbnb: Plataforma MLOps sirve 150+ modelos a 1M+ predicciones/segundo

#### MLOps vs DevOps Tradicional (5 minutos)
| Aspecto | DevOps Tradicional | MLOps |
|---------|-------------------|-------|
| **Artefactos** | Código | Código + Datos + Modelos |
| **Testing** | Tests unitarios | Tests de datos + Tests de modelos |
| **Despliegue** | Blue/green | Pruebas A/B + Canary |
| **Monitoreo** | Métricas del sistema | Rendimiento del modelo + deriva |
| **Rollback** | Versión anterior del código | Modelo anterior + reentrenamiento |

### Discusión Interactiva (5 minutos)
**Preguntas para estudiantes:**
1. "¿En qué proyectos ML han trabajado? ¿Qué desafíos enfrentaron para llevarlos a producción?"
2. "En sus proyectos actuales/pasados, ¿cómo rastrean experimentos?"
3. "¿Cómo saben si su modelo desplegado sigue funcionando correctamente?"

---

## 🕐 Parte 2: Visión General del Ecosistema MLFlow (45 minutos)

### Introducción a MLFlow (10 minutos)
**¿Qué es MLFlow?**
MLFlow es una plataforma open-source para gestionar el ciclo de vida completo de machine learning, incluyendo experimentación, reproducibilidad, despliegue y un registro central de modelos.

**¿Por qué MLFlow?**
- **Open Source** - Sin vendor lock-in, comunidad activa
- **Agnóstico al Lenguaje** - Soporte para Python, R, Java, Scala
- **Agnóstico al Framework** - Funciona con scikit-learn, TensorFlow, PyTorch, etc.
- **Agnóstico a la Nube** - Ejecuta en AWS, Azure, GCP, o on-premises

### Profundización en Componentes MLFlow (25 minutos)

#### 1. MLFlow Tracking (8 minutos)
**Propósito:** Registrar y consultar experimentos (código, datos, configuración, resultados)

**Conceptos Clave:**
- **Experimento:** Grupo nombrado de ejecuciones (ej. "customer-churn-model")
- **Run:** Ejecución única de código ML
- **Parámetros:** Valores de entrada (hiperparámetros, configuración)
- **Métricas:** Valores numéricos a optimizar (precisión, RMSE)
- **Artefactos:** Archivos de salida (modelos, gráficos, datos)

**Demo en Vivo:** Mostrar UI de MLFlow con experimento de ejemplo

#### 2. MLFlow Projects (7 minutos)
**Propósito:** Empaquetar código ML para ejecuciones reproducibles

**Características Clave:**
- **Archivo MLproject:** Define puntos de entrada y dependencias
- **Gestión de entornos:** Conda, Docker, virtualenv
- **Ejecución remota:** Ejecutar en diferentes entornos
- **Paso de parámetros:** Interfaz de línea de comandos

**Ejemplo de archivo MLproject:**
```yaml
name: My ML Project
conda_env: conda.yaml
entry_points:
  main:
    parameters:
      alpha: {type: float, default: 0.5}
    command: "python train.py --alpha {alpha}"
```

#### 3. MLFlow Models (5 minutos)
**Propósito:** Formato estándar para empaquetar modelos ML

**Beneficios Clave:**
- **Múltiples sabores:** scikit-learn, tensorflow, pytorch, etc.
- **Interfaz unificada:** Misma API de predicción sin importar el framework
- **Listo para despliegue:** Puede desplegarse en varias plataformas
- **Metadatos:** Firma del modelo, requisitos, esquema entrada/salida

#### 4. MLFlow Model Registry (5 minutos)
**Propósito:** Almacén central para gestionar el ciclo de vida del modelo

**Características Clave:**
- **Versionado de modelos:** Rastrear evolución del modelo
- **Transiciones de etapa:** Development → Staging → Production
- **Flujos de aprobación:** Controlar promoción de modelos
- **Seguimiento de linaje:** Conectar modelos con experimentos

### Arquitectura y Despliegue MLFlow (10 minutos)

#### Patrones de Despliegue
1. **Desarrollo Local:**
   ```bash
   mlflow ui  # Ejecuta en localhost:5000
   ```

2. **Configuración de Equipo:**
   ```bash
   mlflow server --backend-store-uri sqlite:///mlflow.db \
                 --default-artifact-root ./artifacts \
                 --host 0.0.0.0 --port 5000
   ```

3. **Configuración de Producción:**
   ```bash
   mlflow server --backend-store-uri postgresql://user:pass@host/db \
                 --default-artifact-root s3://bucket/path \
                 --host 0.0.0.0 --port 5000
   ```

#### Ecosistema de Integración
- **Plataformas Cloud:** AWS SageMaker, Azure ML, GCP Vertex AI
- **Orquestación:** Apache Airflow, Kubeflow, Prefect
- **Monitoreo:** Prometheus, Grafana, DataDog
- **Almacenamiento:** S3, Azure Blob, GCS, HDFS

---

## 🕐 Parte 3: Mejores Prácticas del Entorno de Desarrollo (15 minutos)

### Gestión Moderna de Entornos Python (8 minutos)
**Por qué la Gestión de Entornos es Importante:**
- Conflictos de dependencias entre proyectos
- Reproducibilidad entre miembros del equipo
- Seguridad (evitar instalación global de paquetes)
- Consistencia de versiones

**Evolución de Herramientas Python:**
- **Forma antigua:** `pip install` globalmente (💀 infierno de dependencias)
- **Mejor:** `virtualenv` + `pip` (gestión manual)
- **Bueno:** `conda` (pesado pero integral)
- **Moderno:** `uv` (rápido, simple, moderno)

**¿Por qué uv?**
- **10-100x más rápido** que pip
- **Basado en Rust** para velocidad y confiabilidad
- **Sintaxis simple** fácil de aprender
- **Estándares modernos** (sigue estándares PEP)
- **Amigable con Docker** para despliegue

### Control de Versiones para Proyectos ML (4 minutos)
**Desafíos Específicos de ML en Git:**
- Datasets grandes no pueden ir en Git
- Notebooks Jupyter crean diffs desordenados
- Archivos de modelos son binarios y grandes
- Resultados de experimentos necesitan seguimiento

**Mejores Prácticas:**
- `.gitignore` para directorios data/ y models/
- Hacer commit de notebooks con outputs limpiados
- Usar MLFlow para resultados de experimentos, no Git
- Separar versionado de datos/modelos del código

### Principios de Reproducibilidad (3 minutos)
**La Pirámide de Reproducibilidad:**
1. **Entorno** - Mismos paquetes, mismas versiones
2. **Datos** - Mismos datasets, mismo preprocesamiento
3. **Código** - Mismos algoritmos, mismos parámetros
4. **Infraestructura** - Mismo hardware, mismo OS

**Herramientas para Cada Nivel:**
- Entorno: `uv`, Docker
- Datos: Artefactos MLFlow, checksums
- Código: Git, proyectos MLFlow
- Infraestructura: Docker, plantillas cloud

---

## 🛠️ Parte 4: Laboratorio Práctico (2.5 horas)

### Lab 1: Configuración del Entorno con uv (45 minutos)

#### Paso 1: Instalar uv (10 minutos)
**Para que los estudiantes sigan:**

```bash
# Instalar uv (gestor rápido de paquetes Python)
# macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows:
powershell -c "irm https://astral.sh/uv/install.sh | iex"

# Verificar instalación
uv --version
```

**Solución de Problemas:**
- Si curl falla: Descargar desde https://github.com/astral-sh/uv/releases
- Si Windows security bloquea: Usar `--execution-policy bypass`
- Si detrás de firewall corporativo: Usar método de instalación alternativo

#### Paso 2: Crear Estructura del Proyecto (15 minutos)
```bash
# Crear proyecto del curso MLOps
uv init mlops-course
cd mlops-course

# Examinar la estructura generada
ls -la
cat pyproject.toml
```

**Explicar pyproject.toml:**
```toml
[project]
name = "mlops-course"
version = "0.1.0"
description = "Curso MLOps usando MLFlow"
authors = [
    {name = "Tu Nombre", email = "tu.email@ejemplo.com"}
]
dependencies = []

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

#### Paso 3: Agregar Dependencias (10 minutos)
```bash
# Agregar dependencias ML centrales
uv add mlflow[extras] scikit-learn pandas numpy matplotlib seaborn

# Agregar dependencias API
uv add fastapi uvicorn python-dotenv

# Agregar herramientas de desarrollo (grupo separado)
uv add pytest black ruff mypy pre-commit --group dev

# Agregar librerías ML opcionales
uv add optuna sentence-transformers --group ml

# Activar entorno
uv sync
source .venv/bin/activate  # Linux/Mac
# o .venv\Scripts\activate  # Windows
```

**Verificar instalación:**
```bash
python -c "import mlflow; print(mlflow.__version__)"
mlflow --version
```

#### Paso 4: Configuración de Estructura del Proyecto (10 minutos)
```bash
# Crear estructura de proyecto profesional
mkdir -p src/{data,models,evaluation,utils}
mkdir -p tests/{unit,integration}
mkdir -p notebooks
mkdir -p docker
mkdir -p .github/workflows

# Crear archivos __init__.py
touch src/__init__.py
touch src/data/__init__.py
touch src/models/__init__.py
touch src/evaluation/__init__.py
touch src/utils/__init__.py
```

**Crear .gitignore:**
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Entornos virtuales
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# MLFlow
mlruns/
mlartifacts/

# Datos
data/raw/
data/processed/
*.csv
*.xlsx
*.json
*.pkl

# Modelos
models/
*.joblib
*.pickle

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

### Lab 2: Configuración del Proyecto e Integración Git (30 minutos)

#### Paso 1: Configuración del Repositorio Git (10 minutos)
```bash
# Inicializar repositorio Git
git init
git add .gitignore
git commit -m "Initial commit: project structure"

# Conectar a GitHub (los estudiantes deben crear repo primero)
git remote add origin https://github.com/USERNAME/mlops-course.git
git branch -M main
git push -u origin main
```

#### Paso 2: Configuración del Entorno (10 minutos)
**Crear archivo .env:**
```bash
# Crear configuración de entorno
cat > .env << EOF
# Configuración MLFlow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=iris-classification

# Configuración Base de Datos (para uso posterior)
DATABASE_URL=sqlite:///mlflow.db

# Configuración API
API_HOST=0.0.0.0
API_PORT=8000

# Configuraciones de Desarrollo
DEBUG=True
LOG_LEVEL=INFO
EOF
```

**Crear src/utils/config.py:**
```python
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "default")
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///mlflow.db")
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

config = Config()
```

#### Paso 3: Configuración del Servidor MLFlow (10 minutos)
```bash
# Iniciar MLFlow UI en segundo plano
mlflow ui --host 0.0.0.0 --port 5000 &

# Verificar que está ejecutándose
curl http://localhost:5000

# Abrir en navegador
echo "MLFlow UI: http://localhost:5000"
```

**Mostrar a los estudiantes la UI de MLFlow:**
- Navegar a http://localhost:5000
- Explicar la interfaz: Experiments, Runs, Models
- Mostrar estado vacío antes de crear experimentos

### Lab 3: Primer Experimento MLFlow (75 minutos)

#### Paso 1: Crear Dataset de Muestra (15 minutos)
**Crear src/data/load_data.py:**
```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from typing import Tuple
import os

def load_iris_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Cargar y retornar dataset iris como objetos pandas."""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')
    return X, y

def load_wine_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Cargar y retornar dataset wine como objetos pandas."""
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target, name='target')
    return X, y

def get_train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """Dividir datos en conjuntos de entrenamiento y prueba."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def save_data(X_train, X_test, y_train, y_test, data_dir: str = "data"):
    """Guardar divisiones de entrenamiento/prueba en archivos CSV."""
    os.makedirs(data_dir, exist_ok=True)
    
    # Guardar datos de entrenamiento
    train_data = X_train.copy()
    train_data['target'] = y_train
    train_data.to_csv(f"{data_dir}/train.csv", index=False)
    
    # Guardar datos de prueba
    test_data = X_test.copy()
    test_data['target'] = y_test
    test_data.to_csv(f"{data_dir}/test.csv", index=False)
    
    print(f"Datos guardados en {data_dir}/")
    print(f"Forma train: {train_data.shape}")
    print(f"Forma test: {test_data.shape}")

if __name__ == "__main__":
    # Cargar datos
    X, y = load_iris_data()
    
    # Dividir datos
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)
    
    # Guardar datos
    save_data(X_train, X_test, y_train, y_test)
```

**Ejecutar preparación de datos:**
```bash
python src/data/load_data.py
ls data/
head data/train.csv
```

#### Paso 2: Crear Pipeline ML Básico (25 minutos)
**Crear src/models/train.py:**
```python
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.config import config
import argparse
import os

def load_data():
    """Cargar datos de entrenamiento y prueba."""
    train_data = pd.read_csv("data/train.csv")
    test_data = pd.read_csv("data/test.csv")
    
    X_train = train_data.drop('target', axis=1)
    y_train = train_data['target']
    X_test = test_data.drop('target', axis=1)
    y_test = test_data['target']
    
    return X_train, X_test, y_train, y_test

def train_model(model_type: str = "random_forest", **model_params):
    """Entrenar un modelo con seguimiento MLFlow."""
    
    # Configurar URI de seguimiento MLFlow y experimento
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    
    with mlflow.start_run() as run:
        # Registrar información de la ejecución
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("developer", "estudiante")
        mlflow.set_tag("purpose", "entrenamiento")
        
        # Cargar datos
        X_train, X_test, y_train, y_test = load_data()
        
        # Registrar información de datos
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("features", X_train.shape[1])
        mlflow.log_param("classes", len(np.unique(y_train)))
        
        # Crear modelo
        if model_type == "random_forest":
            model = RandomForestClassifier(**model_params)
        elif model_type == "logistic_regression":
            model = LogisticRegression(**model_params)
        else:
            raise ValueError(f"Tipo de modelo desconocido: {model_type}")
        
        # Registrar parámetros del modelo
        for param, value in model_params.items():
            mlflow.log_param(param, value)
        
        # Entrenar modelo
        model.fit(X_train, y_train)
        
        # Hacer predicciones
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calcular métricas
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        test_precision = precision_score(y_test, y_pred_test, average='weighted')
        test_recall = recall_score(y_test, y_pred_test, average='weighted')
        test_f1 = f1_score(y_test, y_pred_test, average='weighted')
        
        # Registrar métricas
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1", test_f1)
        
        # Crear y guardar gráficos
        create_confusion_matrix_plot(y_test, y_pred_test)
        create_feature_importance_plot(model, X_train.columns)
        
        # Registrar modelo
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name=f"iris-{model_type}"
        )
        
        # Imprimir resultados
        print(f"Run ID: {run.info.run_id}")
        print(f"Modelo: {model_type}")
        print(f"Precisión Test: {test_accuracy:.4f}")
        print(f"MLFlow UI: {config.MLFLOW_TRACKING_URI}")
        
        return model, run.info.run_id

def create_confusion_matrix_plot(y_true, y_pred):
    """Crear y guardar gráfico de matriz de confusión."""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png", "plots")
    plt.close()

def create_feature_importance_plot(model, feature_names):
    """Crear y guardar gráfico de importancia de características."""
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.title('Importancia de Características')
        plt.xlabel('Importancia')
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png", "plots")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", default="random_forest", choices=["random_forest", "logistic_regression"])
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    
    args = parser.parse_args()
    
    model_params = {
        "random_state": args.random_state
    }
    
    if args.model_type == "random_forest":
        model_params.update({
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth
        })
    
    model, run_id = train_model(args.model_type, **model_params)
    print(f"¡Entrenamiento completado! Verificar MLFlow UI: {config.MLFLOW_TRACKING_URI}")
```

#### Paso 3: Ejecutar Primeros Experimentos (20 minutos)
```bash
# Ejecutar primer experimento con parámetros por defecto
python src/models/train.py

# Ejecutar experimento con diferentes parámetros
python src/models/train.py --model-type random_forest --n-estimators 50 --max-depth 5

# Ejecutar regresión logística
python src/models/train.py --model-type logistic_regression

# Ejecutar múltiples experimentos con diferentes parámetros
python src/models/train.py --n-estimators 200 --max-depth 10
python src/models/train.py --n-estimators 50 --max-depth 3
```

#### Paso 4: Explorar MLFlow UI (15 minutos)
**Guiar a los estudiantes a través de MLFlow UI:**

1. **Navegar a http://localhost:5000**
2. **Explorar Experimentos:**
   - Hacer clic en el nombre del experimento
   - Comparar diferentes ejecuciones
   - Ordenar por métricas

3. **Examinar Ejecuciones Individuales:**
   - Hacer clic en un run ID
   - Revisar parámetros, métricas, artefactos
   - Descargar artefactos

4. **Comparar Ejecuciones:**
   - Seleccionar múltiples ejecuciones
   - Hacer clic en botón "Compare"
   - Analizar relaciones parámetro vs métrica

5. **Model Registry:**
   - Navegar a pestaña "Models"
   - Ver modelos registrados
   - Explorar versiones de modelos

**Preguntas de Discusión:**
- ¿Cuál modelo tuvo mejor rendimiento y por qué?
- ¿Cómo afectan diferentes parámetros al rendimiento?
- ¿Qué información es más útil para debugging?

---

## 🎯 Cierre de Sesión (10 minutos)

### Revisión de Logros Clave (5 minutos)
**Lo que logramos:**
1. ✅ Entendimos fundamentos MLOps y valor empresarial
2. ✅ Configuramos entorno de desarrollo profesional con uv
3. ✅ Creamos primeros experimentos MLFlow con seguimiento apropiado
4. ✅ Exploramos MLFlow UI para comparación de experimentos
5. ✅ Establecimos estructura de proyecto y mejores prácticas

### Vista Previa de la Siguiente Sesión (3 minutos)
**Vista Previa Sesión 2:**
- MLFlow Model Registry Avanzado
- Selección estratégica de herramientas (marco de decisión MLFlow vs DVC)
- Estrategias integrales de testing ML
- Flujos de promoción de modelos

### Q&A y Solución de Problemas (2 minutos)
**Problemas Comunes:**
- MLFlow UI no carga → Verificar puerto 5000, reiniciar servidor
- Problemas instalación uv → Métodos de instalación alternativos
- Errores de importación → Verificar activación de entorno virtual

---

## 📋 Checkpoint de Evaluación

### Preguntas de Verificación de Conocimiento:
1. ¿Cuáles son los cuatro componentes principales de MLFlow?
2. ¿Cuándo elegiría artefactos MLFlow sobre DVC para versionado de datos?
3. ¿Cuál es la diferencia entre parámetros y métricas en MLFlow?
4. ¿Cómo asegurar reproducibilidad en experimentos ML?

### Verificación Práctica:
- [ ] Estudiante tiene entorno uv funcionando
- [ ] MLFlow UI accesible y mostrando experimentos
- [ ] Al menos 3 ejecuciones de experimentos registradas
- [ ] Estructura de proyecto sigue mejores prácticas
- [ ] Repositorio Git creado y comprometido

---

## 📎 Recursos Adicionales

### Para Estudiantes:
- [Documentación MLFlow](https://mlflow.org/docs/latest/)
- [Documentación uv](https://docs.astral.sh/uv/)
- [Mejores Prácticas Git para ML](https://dvc.org/doc/user-guide/how-to/data-management/git-best-practices)

### Para Instructores:
- Soluciones de problemas comunes
- Ejercicios extendidos para estudiantes avanzados
- Datasets alternativos para variedad

---

## 🔧 Guía de Solución de Problemas

### Problemas Comunes de Estudiantes:

#### 1. Instalación de uv Falla
**Síntomas:** Command not found, errores SSL
**Soluciones:**
- Usar método de instalación alternativo
- Verificar configuraciones de firewall corporativo
- Descargar binario manualmente desde releases de GitHub

#### 2. MLFlow UI No Inicia
**Síntomas:** Puerto 5000 ya en uso, conexión rechazada
**Soluciones:**
```bash
# Verificar qué está usando puerto 5000
lsof -i :5000  # macOS/Linux
netstat -ano | findstr 5000  # Windows

# Usar puerto diferente
mlflow ui --port 5001

# Matar proceso existente
kill -9 [PID]
```

#### 3. Errores de Importación
**Síntomas:** ModuleNotFoundError
**Soluciones:**
```bash
# Verificar que entorno virtual está activado
which python
echo $VIRTUAL_ENV

# Reinstalar dependencias
uv sync --reinstall

# Verificar ruta Python
python -c "import sys; print(sys.path)"
```

#### 4. Problemas del Repositorio Git
**Síntomas:** Permission denied, authentication failed
**Soluciones:**
- Configurar SSH keys para GitHub
- Usar personal access tokens
- Configurar credenciales Git apropiadamente

Este plan de sesión detallado proporciona todo lo necesario para enseñar exitosamente la Sesión 1, con timing claro, ejercicios prácticos y guía de solución de problemas.