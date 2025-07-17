# MLOps con MLFlow: Curso de Cero a Héroe

**🌍 Idiomas:** [English](README.md) | [Español](README-ES.md)

**Duración:** 20 horas (5 sesiones × 4 horas)  
**Formato:** Teórico + Práctico  
**Stack Tecnológico:** Python, MLFlow (Última Versión), Docker, Git

## 🎯 Resumen del Curso

Este curso integral enseña los fundamentos de MLOps utilizando MLFlow como plataforma principal. Los estudiantes aprenderán a gestionar el ciclo de vida completo de ML desde la experimentación hasta el despliegue en producción utilizando las mejores prácticas de la industria.

## 📋 Prerrequisitos

- Programación básica en Python (pandas, scikit-learn)
- Comprensión de conceptos de machine learning
- Familiaridad con control de versiones Git
- Conocimiento básico de línea de comandos
- Fundamentos de Docker (contenedores, imágenes, Dockerfile)

## 🎓 Objetivos de Aprendizaje

Al finalizar este curso, los estudiantes serán capaces de:
- Implementar pipelines MLOps end-to-end usando MLFlow
- Rastrear experimentos, gestionar modelos y desplegar aplicaciones ML
- Aplicar mejores prácticas de MLOps para flujos de trabajo GenAI
- Construir sistemas ML listos para producción con monitoreo y testing apropiados
- Colaborar efectivamente en equipos ML usando MLFlow y herramientas modernas
- Controlar versiones de datos y modelos usando DVC
- Implementar pipelines CI/CD para proyectos ML
- Aplicar mejores prácticas de seguridad y gobernanza en MLOps

---

## 📅 Estructura del Curso y Sesiones Detalladas

### Resumen de Sesiones
| Sesión | Tema | Duración | Materiales |
|---------|-------|----------|-----------|
| **1** | Fundamentos MLOps y Configuración del Entorno | 4 horas | [📋 Plan de Sesión Detallado](session-1-detailed.md) |
| **2** | Características Avanzadas MLFlow y Estrategia de Selección de Herramientas | 4 horas | Próximamente |
| **3** | CI/CD, Seguridad y Despliegue en Producción | 4 horas | Próximamente |
| **4** | MLOps GenAI y Optimización de Costos | 4 horas | Próximamente |
| **5** | Operaciones de Producción y MLOps Empresarial | 4 horas | Próximamente |

---

## 📅 Sesión 1: Fundamentos MLOps y Configuración del Entorno (4 horas)
*[👉 Ver Plan de Enseñanza Detallado](session-1-detailed.md)*

### Teoría (1.5 horas)
- **Introducción a MLOps** (30 min)
  - Qué es MLOps y por qué es importante
  - Desafíos del ciclo de vida ML vs software tradicional
  - Niveles de madurez MLOps y estrategias de adopción
  - ROI e impacto comercial de MLOps
  
- **Visión General del Ecosistema MLFlow** (45 min)
  - Componentes MLFlow: Tracking, Projects, Models, Registry
  - Opciones de arquitectura y despliegue
  - Integración con plataformas cloud y otras herramientas
  - Comparación MLFlow vs alternativas
  
- **Mejores Prácticas del Entorno de Desarrollo** (15 min)
  - Gestión moderna de entornos Python
  - Estrategias de control de versiones para proyectos ML
  - Principios de reproducibilidad y desafíos

### Laboratorio Práctico (2.5 horas)
- **Configuración del Entorno con uv** (45 min)
  ```bash
  # Instalar uv (gestor rápido de paquetes Python)
  curl -LsSf https://astral.sh/uv/install.sh | sh
  
  # Crear proyecto del curso MLOps
  uv init mlops-course
  cd mlops-course
  
  # Agregar todas las dependencias requeridas
  uv add mlflow[extras] scikit-learn pandas numpy matplotlib seaborn
  uv add pytest python-dotenv fastapi uvicorn --group dev
  uv add pre-commit black ruff mypy --group dev
  
  # Activar entorno y verificar
  uv sync
  source .venv/bin/activate
  mlflow --version
  
  # Iniciar MLFlow UI
  mlflow ui --host 0.0.0.0 --port 5000
  ```

- **Estructura del Proyecto y Configuración Git** (30 min)
  ```
  mlops-course/
  ├── .env                    # Variables de entorno
  ├── .gitignore             # Reglas de Git ignore
  ├── pyproject.toml         # Configuración del proyecto uv
  ├── README.md              # Documentación del proyecto
  ├── src/
  │   ├── __init__.py
  │   ├── data/              # Módulos de procesamiento de datos
  │   ├── models/            # Módulos de entrenamiento de modelos
  │   ├── evaluation/        # Evaluación de modelos
  │   └── utils/             # Funciones de utilidad
  ├── notebooks/             # Notebooks Jupyter
  ├── tests/                 # Pruebas unitarias
  ├── docker/                # Configuraciones Docker
  └── .github/
      └── workflows/         # GitHub Actions
  ```

- **Primer Experimento MLFlow** (75 min)
  - Crear un pipeline ML integral con scikit-learn
  - Registrar parámetros, métricas, artefactos y metadatos del modelo
  - Explorar MLFlow UI en profundidad
  - Comparar múltiples ejecuciones y analizar resultados
  - Implementar validación básica de datos y testing de modelos

### 🛠️ Proyecto Práctico
Crear un modelo básico de clasificación con seguimiento MLFlow:
```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Iniciar experimento MLFlow
mlflow.set_experiment("iris-classification")

with mlflow.start_run():
    # Cargar datos
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Entrenar modelo
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Registrar parámetros y métricas
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy_score(y_test, model.predict(X_test)))
    
    # Registrar modelo
    mlflow.sklearn.log_model(model, "model")
```

---

## 📅 Sesión 2: Características Avanzadas MLFlow y Estrategia de Selección de Herramientas (4 horas)

### Teoría (1.5 horas)
- **Profundización en MLFlow Model Registry** (30 min)
  - Estrategias de gestión del ciclo de vida del modelo
  - Entornos de staging (Development, Staging, Production)
  - Flujos de trabajo de promoción de modelos y procesos de aprobación
  - Linaje de modelos y gestión de metadatos

- **Estrategia de Gestión de Datos: MLFlow vs DVC** (45 min)
  - **Cuándo los artefactos MLFlow son suficientes:**
    - Datasets < 500MB
    - Pipelines de preprocesamiento simples
    - Equipos pequeños (< 5 científicos de datos)
    - Entornos de prototipado y aprendizaje
  
  - **Cuándo DVC se vuelve necesario:**
    - Datasets grandes (> 1GB)
    - Pipelines de datos complejos multi-etapa
    - Múltiples equipos compartiendo datasets
    - Organizaciones con datos pesados y actualizaciones frecuentes
  
  - **Marco de Decisión:**
    | Criterio | Artefactos MLFlow | DVC + MLFlow | Ambos |
    |----------|------------------|--------------|-------|
    | **Tamaño de Datos** | < 500MB | > 1GB | Mixto |
    | **Complejidad Pipeline** | Transformaciones simples | Pipelines multi-etapa | Varía |
    | **Tamaño Equipo** | 1-5 personas | 5+ personas | Organizaciones grandes |
    | **Costo Almacenamiento** | Poca preocupación | Alta preocupación | Crítico |
    | **Curva Aprendizaje** | Mínima | Moderada | Compleja |
    
  - **Ejemplos del mundo real y casos de estudio**
  - **Estrategias de migración**: Comenzar con MLFlow, cuándo agregar DVC

- **Estrategias de Testing para ML** (15 min)
  - Testing unitario para código ML vs software tradicional
  - Validación de datos y verificaciones de calidad
  - Estrategias de testing de modelos (rendimiento, equidad, robustez)

### Laboratorio Práctico (2.5 horas)
- **MLFlow Model Registry Avanzado** (75 min)
  - Registrar modelos programáticamente
  - Etapas del modelo: Staging, Production, Archived
  - Versionado y linaje de modelos
  - Integración de webhooks para CI/CD

- **Ejercicio de Selección de Herramientas** (30 min)
  - **Análisis de Escenarios**: Los estudiantes reciben 3 escenarios diferentes de empresas
    - Startup con datasets pequeños
    - Empresa mediana con necesidades crecientes de datos
    - Empresa con pipelines de datos complejos
  - **Discusión Grupal**: Qué enfoque (solo MLFlow vs MLFlow + DVC) para cada escenario
  - **Documentación de Decisiones**: Los estudiantes documentan su razonamiento

- **Testing Integral de ML** (75 min)
  - Tests unitarios, de integración y específicos para ML
  - Validación de datos y verificaciones de calidad
  - Estrategias de testing de modelos (rendimiento, equidad, robustez)
  - Automatización de testing con pytest

### 🛠️ Proyecto Práctico
Construir un pipeline de optimización de hiperparámetros:
```python
import mlflow
import optuna
from mlflow.tracking import MlflowClient

def objective(trial):
    with mlflow.start_run(nested=True):
        # Sugerir parámetros
        n_estimators = trial.suggest_int('n_estimators', 10, 100)
        max_depth = trial.suggest_int('max_depth', 1, 10)
        
        # Entrenar y evaluar modelo
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth
        )
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        
        # Registrar en MLFlow
        mlflow.log_params(trial.params)
        mlflow.log_metric("accuracy", accuracy)
        
        return accuracy

# Ejecutar optimización
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
```

---

## 📅 Sesión 3: CI/CD, Seguridad y Despliegue en Producción (4 horas)

### Teoría (1 hora)
- **CI/CD para MLOps** (30 min)
  - Diferencias entre CI/CD de software y ML CI/CD
  - Estrategias de testing automatizado para pipelines ML
  - Compuertas de validación de modelos y procesos de aprobación
  - Estrategias de rollback para modelos ML
  
- **Seguridad y Gobernanza** (30 min)
  - Consideraciones de seguridad del modelo
  - Privacidad de datos y cumplimiento (GDPR, CCPA)
  - Gobernanza de modelos y rastros de auditoría
  - Control de acceso y autenticación en MLFlow

### Laboratorio Práctico (3 horas)
- **GitHub Actions para MLOps** (90 min)
  - Crear pipelines ML automatizados
  - Implementar compuertas de testing y validación
  - Desplegar modelos a staging y producción
  - Escaneo de seguridad y verificaciones de cumplimiento

- **Despliegue Docker Seguro** (90 min)
  - Builds Docker multi-etapa para aplicaciones ML
  - Mejores prácticas de seguridad para modelos containerizados
  - Gestión de entornos y manejo de secretos
  - Patrones de despliegue en producción

### 🛠️ Proyecto Práctico
Crear un proyecto MLFlow completo con despliegue:

**Archivo MLproject:**
```yaml
name: ml-pipeline

conda_env: conda.yaml

entry_points:
  main:
    parameters:
      max_depth: {type: int, default: 5}
      n_estimators: {type: int, default: 100}
    command: "python train.py --max-depth {max_depth} --n-estimators {n_estimators}"
  
  serve:
    command: "mlflow models serve -m models:/iris-model/Production -p 1234"
```

**Despliegue:**
```bash
# Construir y servir modelo
mlflow models build-docker -m "models:/iris-model/1" -n "iris-model"
docker run -p 5000:8080 iris-model
```

---

## 📅 Sesión 4: MLOps GenAI y Optimización de Costos (4 horas)

### Teoría (1 hora)
- **Desafíos MLOps GenAI** (30 min)
  - Complejidades de gestión del ciclo de vida LLM
  - Estrategias de ingeniería de prompts y versionado
  - Métricas de evaluación para GenAI (BLEU, ROUGE, métricas personalizadas)
  - Trade-offs entre fine-tuning vs RAG vs ingeniería de prompts
  
- **Gestión y Optimización de Costos** (30 min)
  - Monitoreo de costos LLM y presupuestación
  - Técnicas de optimización de uso de tokens
  - Selección de modelos basada en costo-rendimiento
  - Estrategias de caché para respuestas LLM

### Laboratorio Práctico (3 horas)
- **Seguimiento de Experimentos LLM y Control de Costos** (90 min)
  - Rastrear experimentos LLM con MLFlow
  - Implementar monitoreo de costos y presupuestación
  - Gestión y versionado de plantillas de prompts
  - Métricas de evaluación personalizadas para GenAI

- **Pipeline GenAI Avanzado** (90 min)
  - Construir sistemas RAG (Retrieval Augmented Generation)
  - Implementar pruebas A/B para aplicaciones GenAI
  - Comparación y selección de múltiples modelos
  - Estrategias de optimización y caché de respuestas

### 🛠️ Proyecto Práctico
Construir un pipeline GenAI listo para producción:
```python
# Sistema RAG completo con seguimiento MLFlow
import mlflow
from mlflow.models import infer_signature
import chromadb
from sentence_transformers import SentenceTransformer

class RAGPipeline:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.vectordb = chromadb.Client()
        
    def track_experiment(self, query, context, response):
        with mlflow.start_run():
            # Registrar componentes del prompt
            mlflow.log_text(query, "user_query.txt")
            mlflow.log_text(context, "retrieved_context.txt")
            mlflow.log_text(response, "generated_response.txt")
            
            # Registrar métricas
            mlflow.log_metric("context_relevance", self.calculate_relevance(query, context))
            mlflow.log_metric("response_quality", self.evaluate_response(response))
            
            # Registrar costos y rendimiento
            mlflow.log_metric("retrieval_time", self.retrieval_time)
            mlflow.log_metric("generation_cost", self.generation_cost)
```

---

## 📅 Sesión 5: Operaciones de Producción y MLOps Empresarial (4 horas)

### Teoría (1 hora)
- **Debugging y Resolución de Problemas en Producción** (30 min)
  - Problemas comunes de producción y soluciones
  - Debugging de degradación del rendimiento de modelos ML
  - Análisis de logs y seguimiento de errores
  - Respuesta a incidentes para sistemas ML
  
- **Colaboración en Equipo y Gobernanza** (30 min)
  - Estructuras de equipos MLOps y responsabilidades
  - Marcos de gobernanza de modelos
  - Requisitos de cumplimiento y auditoría
  - Gestión del cambio para sistemas ML

### Laboratorio Práctico (3 horas)
- **Monitoreo y Alertas de Producción** (90 min)
  - Implementar monitoreo de rendimiento de modelos
  - Configurar sistemas de alertas automatizadas
  - Monitorear deriva de datos y degradación de modelos
  - Crear dashboards para comunicación con stakeholders

- **Integración Empresarial y Gobernanza** (90 min)
  - Configuración de control de acceso basado en roles (RBAC)
  - Flujos de trabajo de aprobación de modelos
  - Reportes de cumplimiento y rastros de auditoría
  - Integración con herramientas empresariales

### 🛠️ Proyecto Capstone Final
Sistema MLOps completo de nivel empresarial:

**Requisitos del Proyecto:**
- Pipeline ML end-to-end con gestión de entorno uv
- Suite de testing integral (unitarias, integración, rendimiento)
- CI/CD con GitHub Actions
- Despliegue Docker seguro
- Monitoreo y alertas de producción
- Seguimiento y optimización de costos
- Características de colaboración en equipo
- Documentación de gobernanza y cumplimiento

---

## 📊 Evaluación y Valoración

### Asignaciones Individuales (50%)
- Experimentos y laboratorios de sesiones (30%)
- Calidad de código y documentación (10%)
- Implementación de mejores prácticas MLFlow (10%)

### Proyecto Grupal y Presentación (50%)
- **Desarrollo de pipeline MLOps end-to-end (30%)**
  - Colaboración en equipo usando MLFlow
  - Implementación de despliegue en producción
  - Calidad de código y documentación
  
- **Presentación y Demo del Equipo (20%)**
  - Presentación de 15 minutos por equipo
  - Demostración en vivo del pipeline MLOps
  - Sesión de Q&A con instructores y compañeros
  - Profundidad técnica y claridad de explicación

### Guías de Presentación
- **Duración**: 15 minutos de presentación + 5 minutos Q&A
- **Formato**: Demo en vivo con soporte de slides
- **Requisitos de Contenido**:
  - Planteamiento del problema y enfoque de solución
  - Recorrido de implementación MLFlow
  - Demostración en vivo del pipeline
  - Desafíos enfrentados y soluciones implementadas
  - Lecciones aprendidas y mejores prácticas
- **Criterios de Evaluación**:
  - Precisión técnica y calidad de implementación
  - Claridad de presentación y comunicación
  - Demostración de colaboración en equipo
  - Innovación y aplicación de mejores prácticas

---

## 🛠️ Stack Tecnológico y Herramientas

### Herramientas MLOps Centrales (Open Source)
- **MLFlow** - Seguimiento de experimentos, registro de modelos, despliegue
- **Python 3.8+** - Lenguaje de programación principal
- **GitHub** - Control de versiones y colaboración
- **GitHub Actions** - Automatización CI/CD
- **Docker** - Containerización y despliegue

### Herramientas de Soporte
- **pytest** - Testing unitario para código ML
- **FastAPI** - APIs personalizadas para servir modelos
- **python-dotenv** - Configuración de entorno
- **pre-commit** - Automatización de calidad de código
- **SQLite/PostgreSQL** - Almacenamiento backend MLFlow

### Entorno de Desarrollo
- **uv** - Gestor rápido de paquetes y entornos Python
- **VS Code** (IDE recomendado)
- **Git** - Cliente de control de versiones
- **Docker Desktop** - Gestión de contenedores

### Configuración del Entorno
```bash
# Instalar uv (gestor rápido de paquetes Python)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Crear proyecto del curso
uv init mlops-course
cd mlops-course

# Agregar dependencias centrales
uv add mlflow[extras] scikit-learn pandas numpy matplotlib seaborn
uv add fastapi uvicorn python-dotenv

# Agregar herramientas de desarrollo
uv add pytest black ruff mypy pre-commit --group dev

# Agregar librerías ML opcionales
uv add optuna sentence-transformers --group ml

# Activar entorno
uv sync
source .venv/bin/activate  # Linux/Mac
# o .venv\Scripts\activate  # Windows
```

## 📚 Recursos Requeridos

### Documentación Oficial
- [Guía ML de MLFlow](https://mlflow.org/docs/latest/ml/)
- [Guía GenAI de MLFlow](https://mlflow.org/docs/latest/genai/)
- [Documentación GitHub Actions](https://docs.github.com/en/actions)

### Datasets de Muestra y APIs
- **ML Tradicional**: Iris, Boston Housing, Titanic, Wine Quality
- **Series Temporales**: Precios de acciones, datos de sensores IoT, pronósticos de ventas
- **NLP/GenAI**: Reseñas de películas, tickets de soporte al cliente, documentación
- **Visión Computacional**: CIFAR-10, Fashion-MNIST (si el tiempo lo permite)

### Recursos Cloud (Opcionales)
- **Niveles gratuitos**: AWS, Google Cloud, Azure para demos de despliegue
- **Alternativas**: Contenedores Docker locales para experiencia completamente offline

---

## 🔧 Temas Adicionales Más Allá de la Documentación MLFlow

Este curso cubre temas esenciales de MLOps que complementan la documentación oficial de MLFlow:

### **Ingeniería y Calidad de Datos**
- **Validación de Datos**: Verificaciones de calidad, perfilado y detección de deriva
- **Gestión de Datasets**: Versionado de datasets dentro de artefactos MLFlow
- **Ingeniería de Características**: Mejores prácticas y reproducibilidad

### **Ingeniería de Software para ML**
- **Estrategias de Testing**: Tests unitarios, de integración y específicos para ML
- **Calidad de Código**: Linting (ruff), verificación de tipos (mypy) y hooks pre-commit
- **Gestión de Entornos**: Herramientas Python modernas con uv

### **DevOps e Infraestructura**
- **CI/CD para ML**: Pipelines automatizados de testing, validación y despliegue
- **Containerización**: Mejores prácticas Docker para aplicaciones ML
- **Seguridad**: Autenticación, autorización y despliegues seguros

### **Operaciones de Producción**
- **Monitoreo**: Rendimiento de modelos, deriva de datos y salud del sistema
- **Gestión de Costos**: Optimización de recursos y seguimiento de presupuesto
- **Respuesta a Incidentes**: Debugging y resolución de problemas de sistemas ML

### **Colaboración en Equipo**
- **Gobernanza**: Flujos de trabajo de aprobación de modelos y cumplimiento
- **Documentación**: Documentación técnica e intercambio de conocimientos
- **Gestión de Proyectos**: Metodologías ágiles para proyectos ML

### **Integración Empresarial**
- **Medición de ROI**: Cuantificar el valor e impacto de MLOps
- **Comunicación con Stakeholders**: Presentar conceptos técnicos a usuarios de negocio
- **Gestión del Cambio**: Estrategias de adopción y transformación organizacional

Estos temas aseguran que los estudiantes obtengan conocimiento integral de MLOps más allá del uso de MLFlow, preparándolos para entornos empresariales del mundo real.

---

## 🎯 Ruta de Aprendizaje Post-Curso

### Temas Intermedios
- MLFlow a escala con Kubernetes
- Desarrollo de plugins personalizados MLFlow
- Integración con Apache Airflow
- Estrategias MLOps multi-cloud

### Especializaciones Avanzadas
- MLOps para Visión Computacional
- ML en tiempo real con datos streaming
- Flujos de trabajo de aprendizaje federado
- Gobernanza y cumplimiento MLOps

---

## 🤝 Soporte y Comunidad

- Foro del curso para Q&A
- Horarios de oficina con instructores
- Recursos de la comunidad MLFlow
- Speakers invitados de la industria

**Certificado:** Al completar exitosamente, los participantes reciben un certificado "MLOps con MLFlow Professional".