# Sesión 2A: Dominio de MLFlow Tracking - Plan de Enseñanza Detallado

**🌍 Idiomas:** [English](session-2a-detailed-EN.md) | [Español](session-2a-detailed-ES.md)

**Duración:** 2 horas  
**Formato:** Teoría (30 min) + Laboratorio Práctico (90 min)  
**Prerrequisitos:** Sesión 1 completada, ambiente configurado, experimentos básicos funcionando

---

## 🎯 Objetivos de Aprendizaje de la Sesión

Al finalizar esta sesión, los estudiantes serán capaces de:
- Organizar experimentos MLFlow de manera profesional con naming conventions
- Implementar nested runs para experimentos complejos y multi-etapa
- Dominar auto-logging avanzado con diferentes frameworks ML
- Crear artefactos personalizados ricos (HTML interactivo, reportes, análisis)
- Gestionar experimentos programáticamente y realizar análisis comparativos

---

## 📚 Preparación Pre-Sesión

### Verificación del Instructor
- [ ] Sesión 1 completada por estudiantes
- [ ] MLFlow ambiente funcionando
- [ ] Experimentos básicos de Sesión 1 existentes
- [ ] Ejemplos de nested runs preparados

### Estado del Estudiante
- [ ] Proyecto mlops-course de Sesión 1 funcionando
- [ ] Al menos 1-2 experimentos básicos en MLFlow
- [ ] MLFlow UI accesible en localhost:5000
- [ ] Entorno uv activado y dependencias instaladas

---

## 🕐 Parte 1: Teoría - Advanced MLFlow Tracking (30 minutos)

### Experiment Organization & Best Practices (10 minutos)

#### Naming Conventions y Taxonomías
**Problema:** Después de 50+ experimentos, ¿cómo encontrar el modelo de fraude con feature engineering v2.3?

**Solución: Hierarchical Naming**
```python
# ❌ Malo
mlflow.set_experiment("test")
mlflow.set_experiment("model2")
mlflow.set_experiment("final_model")

# ✅ Bueno
mlflow.set_experiment("fraud-detection/baseline-models")
mlflow.set_experiment("fraud-detection/feature-engineering-v2")
mlflow.set_experiment("fraud-detection/production-candidates")
```

#### Tag Strategy Profesional
**Tags como Metadata Estructurado:**
```python
mlflow.set_tags({
    "team": "fraud-detection",
    "model_family": "tree_based",
    "data_version": "2024_Q1", 
    "feature_set": "engineered_v2.3",
    "experiment_phase": "hyperparameter_tuning",
    "business_objective": "precision_optimization",
    "deployment_target": "real_time_api"
})
```

### Advanced Tracking Patterns (10 minutos)

#### 1. Nested Runs para Experimentos Complejos
**Casos de Uso:**
- **Hyperparameter Optimization:** Parent = estudio, Children = trials individuales
- **Cross-Validation:** Parent = experimento, Children = folds
- **Pipeline Multi-etapa:** Parent = pipeline completo, Children = etapas
- **A/B Testing:** Parent = test, Children = variantes

**Estructura Típica:**
```
Parent Run: "Random Forest Optimization Study"
├── Child Run: "Trial 1: n_estimators=50, max_depth=3"
├── Child Run: "Trial 2: n_estimators=100, max_depth=5"
├── Child Run: "Trial 3: n_estimators=200, max_depth=10"
└── Child Run: "Trial 4: n_estimators=100, max_depth=None"
```

#### 2. Parent-Child Relationships
**Best Practices:**
- **Parent:** Log summary metrics, best parameters, final artifacts
- **Children:** Log individual trial details, intermediate results
- **Naming:** Consistent naming conventions para fácil navegación

### Auto-logging Deep Dive (10 minutos)

#### Framework Support Matrix
| Framework | Parameters | Metrics | Model | Artifacts | Custom Config |
|-----------|------------|---------|-------|-----------|---------------|
| **scikit-learn** | ✅ All params | ✅ Score metrics | ✅ Full model | ✅ Plots | ✅ Extensive |
| **XGBoost** | ✅ Boost params | ✅ Eval metrics | ✅ Booster | ✅ Trees | ✅ Good |
| **LightGBM** | ✅ LGB params | ✅ Valid metrics | ✅ Model | ✅ Feature imp | ✅ Good |
| **TensorFlow** | ✅ Config | ✅ Training | ✅ SavedModel | ✅ Summaries | ✅ Advanced |
| **PyTorch** | ⚠️ Limited | ✅ Losses | ✅ State dict | ✅ Checkpoints | ⚠️ Manual |

#### Auto-logging Configuration
```python
# Configuración granular
mlflow.sklearn.autolog(
    log_input_examples=True,      # Log sample inputs
    log_model_signatures=True,    # Log input/output schema  
    log_models=True,              # Log trained models
    log_datasets=True,            # Log dataset information
    disable=False,                # Enable/disable globally
    exclusive=False,              # Allow manual logging too
    disable_for_unsupported_versions=False,
    silent=False                  # Show autolog info
)
```

---

## 🛠️ Parte 2: Laboratorio Práctico (90 minutos)

### Lab 1: Experiment Organization & Advanced Tracking (40 minutos)

#### Crear src/experiments/advanced_organization.py:
```python
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
import os

# Agregar src al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.config import config
from src.data.load_data import load_iris_data, get_train_test_split

class ExperimentOrganizer:
    """Clase para organizar experimentos MLFlow de manera profesional."""
    
    def __init__(self, project_name="iris-classification"):
        self.project_name = project_name
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        
        # Preparar datos una vez
        X, y = load_iris_data()
        self.X_train, self.X_test, self.y_train, self.y_test = get_train_test_split(X, y)
        
    def create_experiment_hierarchy(self):
        """Crear jerarquía de experimentos organizada."""
        
        experiments = [
            f"{self.project_name}/01-baseline-models",
            f"{self.project_name}/02-feature-engineering", 
            f"{self.project_name}/03-hyperparameter-optimization",
            f"{self.project_name}/04-production-candidates",
            f"{self.project_name}/05-model-comparison-studies"
        ]
        
        for exp_name in experiments:
            try:
                mlflow.create_experiment(exp_name)
                print(f"✅ Created experiment: {exp_name}")
            except mlflow.exceptions.MlflowException:
                print(f"📁 Experiment already exists: {exp_name}")
    
    def baseline_study(self):
        """Estudio baseline con múltiples algoritmos."""
        
        mlflow.set_experiment(f"{self.project_name}/01-baseline-models")
        
        # Definir modelos baseline
        baseline_models = {
            "logistic_regression": LogisticRegression(random_state=42, max_iter=1000),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        # Parent run para todo el estudio baseline
        with mlflow.start_run(run_name="baseline_study_comparison") as parent_run:
            # Tags del estudio
            mlflow.set_tags({
                "study_type": "baseline_comparison",
                "team": "ml_team",
                "data_version": "v1.0",
                "experiment_phase": "baseline",
                "business_objective": "establish_performance_floor",
                "models_count": len(baseline_models)
            })
            
            # Metadata del estudio
            mlflow.log_params({
                "study_date": datetime.now().isoformat(),
                "dataset_size": len(self.X_train) + len(self.X_test),
                "train_size": len(self.X_train),
                "test_size": len(self.X_test),
                "n_features": self.X_train.shape[1],
                "n_classes": len(np.unique(self.y_train))
            })
            
            results = {}
            
            # Entrenar cada modelo en nested run
            for model_name, model in baseline_models.items():
                with mlflow.start_run(run_name=f"baseline_{model_name}", nested=True):
                    # Tags específicos del modelo
                    mlflow.set_tags({
                        "model_type": model_name,
                        "algorithm_family": str(type(model).__name__),
                        "framework": "scikit-learn",
                        "run_type": "baseline_individual"
                    })
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                    
                    # Entrenamiento final
                    model.fit(self.X_train, self.y_train)
                    
                    # Métricas de evaluación
                    train_score = model.score(self.X_train, self.y_train)
                    test_score = model.score(self.X_test, self.y_test)
                    overfitting_ratio = train_score / test_score
                    
                    # Log métricas
                    mlflow.log_metrics({
                        "cv_accuracy_mean": cv_mean,
                        "cv_accuracy_std": cv_std,
                        "train_accuracy": train_score,
                        "test_accuracy": test_score,
                        "overfitting_ratio": overfitting_ratio
                    })
                    
                    # Log parámetros del modelo
                    mlflow.log_params(model.get_params())
                    
                    # Predicciones y reporte detallado
                    y_pred = model.predict(self.X_test)
                    
                    # Classification report
                    report = classification_report(self.y_test, y_pred, output_dict=True)
                    for class_name, metrics in report.items():
                        if isinstance(metrics, dict):
                            for metric_name, value in metrics.items():
                                if isinstance(value, (int, float)):
                                    mlflow.log_metric(f"class_{class_name}_{metric_name}", value)
                    
                    # Confusion matrix plot
                    plt.figure(figsize=(8, 6))
                    cm = confusion_matrix(self.y_test, y_pred)
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                    plt.title(f'Confusion Matrix - {model_name}')
                    plt.ylabel('True Label')
                    plt.xlabel('Predicted Label')
                    plt.tight_layout()
                    
                    cm_filename = f"confusion_matrix_{model_name}.png"
                    plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
                    mlflow.log_artifact(cm_filename, "evaluation_plots")
                    plt.close()
                    
                    # Log modelo
                    mlflow.sklearn.log_model(model, "model")
                    
                    # Guardar resultados para comparación
                    results[model_name] = {
                        "cv_mean": cv_mean,
                        "cv_std": cv_std,
                        "test_accuracy": test_score,
                        "overfitting_ratio": overfitting_ratio
                    }
                    
                    print(f"✅ {model_name}: CV={cv_mean:.4f}±{cv_std:.4f}, Test={test_score:.4f}")
            
            # En parent run: análisis comparativo
            best_model = max(results, key=lambda x: results[x]["test_accuracy"])
            
            mlflow.log_metrics({
                "best_model_accuracy": results[best_model]["test_accuracy"],
                "accuracy_range": max([r["test_accuracy"] for r in results.values()]) - 
                                min([r["test_accuracy"] for r in results.values()]),
                "mean_accuracy": np.mean([r["test_accuracy"] for r in results.values()])
            })
            
            mlflow.log_params({
                "best_baseline_model": best_model,
                "models_tested": list(results.keys())
            })
            
            # Crear gráfico comparativo
            self._create_comparison_plot(results, "baseline_comparison.png")
            mlflow.log_artifact("baseline_comparison.png", "study_analysis")
            
            print(f"""
            🎉 Baseline study completed!
            
            🏆 Best model: {best_model}
            📊 Best accuracy: {results[best_model]['test_accuracy']:.4f}
            
            🔗 View in MLFlow: {config.MLFLOW_TRACKING_URI}
            """)
            
            return results, parent_run.info.run_id
    
    def hyperparameter_optimization_study(self):
        """Estudio de optimización de hiperparámetros con nested runs."""
        
        mlflow.set_experiment(f"{self.project_name}/03-hyperparameter-optimization")
        
        # Grid de parámetros para Random Forest
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Parent run para optimización completa
        with mlflow.start_run(run_name="random_forest_grid_search") as parent_run:
            # Tags del estudio de optimización
            mlflow.set_tags({
                "study_type": "hyperparameter_optimization",
                "optimization_method": "grid_search",
                "model_family": "random_forest",
                "search_space_size": len(list(ParameterGrid(param_grid))),
                "experiment_phase": "optimization"
            })
            
            # Parámetros del estudio
            mlflow.log_params({
                "cv_folds": 5,
                "optimization_metric": "cv_accuracy",
                "param_combinations": len(list(ParameterGrid(param_grid))),
                "random_state": 42
            })
            
            best_score = 0
            best_params = None
            all_results = []
            
            # Iterar por todas las combinaciones
            for i, params in enumerate(ParameterGrid(param_grid)):
                with mlflow.start_run(run_name=f"grid_trial_{i+1}", nested=True):
                    # Tags del trial individual
                    mlflow.set_tags({
                        "trial_number": i+1,
                        "optimization_method": "grid_search",
                        "model_type": "RandomForestClassifier"
                    })
                    
                    # Crear modelo con parámetros específicos
                    model = RandomForestClassifier(**params, random_state=42)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                    
                    # Entrenamiento final
                    model.fit(self.X_train, self.y_train)
                    test_score = model.score(self.X_test, self.y_test)
                    
                    # Log parámetros y métricas
                    mlflow.log_params(params)
                    mlflow.log_metrics({
                        "cv_accuracy_mean": cv_mean,
                        "cv_accuracy_std": cv_std,
                        "test_accuracy": test_score,
                        "trial_number": i+1
                    })
                    
                    # Guardar modelo si es el mejor hasta ahora
                    if cv_mean > best_score:
                        best_score = cv_mean
                        best_params = params.copy()
                        mlflow.sklearn.log_model(model, "best_model_so_far")
                        mlflow.set_tag("is_best_trial", "true")
                    else:
                        mlflow.set_tag("is_best_trial", "false")
                    
                    # Guardar resultados
                    trial_result = params.copy()
                    trial_result.update({
                        'trial': i+1,
                        'cv_mean': cv_mean,
                        'cv_std': cv_std,
                        'test_accuracy': test_score
                    })
                    all_results.append(trial_result)
                    
                    if (i+1) % 10 == 0:
                        print(f"Completed {i+1} trials. Best CV so far: {best_score:.4f}")
            
            # Log resultados finales en parent run
            mlflow.log_metrics({
                "best_cv_accuracy": best_score,
                "optimization_trials_completed": len(all_results)
            })
            
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            
            # Crear análisis de optimización
            results_df = pd.DataFrame(all_results)
            
            # Análisis de importancia de parámetros
            self._analyze_parameter_importance(results_df)
            
            # Log tabla de resultados
            results_df.to_csv("optimization_results.csv", index=False)
            mlflow.log_artifact("optimization_results.csv", "optimization_analysis")
            
            print(f"""
            🎯 Optimization completed!
            
            📊 Best parameters: {best_params}
            📈 Best CV score: {best_score:.4f}
            🔍 Trials completed: {len(all_results)}
            """)
            
            return best_params, best_score, parent_run.info.run_id
    
    def _create_comparison_plot(self, results, filename):
        """Crear gráfico de comparación de modelos."""
        
        plt.figure(figsize=(12, 8))
        
        models = list(results.keys())
        cv_means = [results[m]["cv_mean"] for m in models]
        cv_stds = [results[m]["cv_std"] for m in models]
        test_accs = [results[m]["test_accuracy"] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        plt.subplot(2, 1, 1)
        plt.bar(x - width/2, cv_means, width, yerr=cv_stds, label='CV Accuracy', alpha=0.8)
        plt.bar(x + width/2, test_accs, width, label='Test Accuracy', alpha=0.8)
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.title('Model Performance Comparison')
        plt.xticks(x, models, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        overfitting_ratios = [results[m]["overfitting_ratio"] for m in models]
        bars = plt.bar(models, overfitting_ratios, alpha=0.7)
        plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='No Overfitting')
        plt.xlabel('Models')
        plt.ylabel('Overfitting Ratio (Train/Test)')
        plt.title('Overfitting Analysis')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Color code overfitting bars
        for bar, ratio in zip(bars, overfitting_ratios):
            if ratio > 1.1:
                bar.set_color('red')
            elif ratio > 1.05:
                bar.set_color('orange')
            else:
                bar.set_color('green')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _analyze_parameter_importance(self, results_df):
        """Analizar importancia de parámetros en optimización."""
        
        # Crear visualización de parameter importance
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. n_estimators vs accuracy
        axes[0, 0].scatter(results_df['n_estimators'], results_df['cv_mean'], alpha=0.7)
        axes[0, 0].set_xlabel('n_estimators')
        axes[0, 0].set_ylabel('CV Accuracy')
        axes[0, 0].set_title('n_estimators vs Performance')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. max_depth vs accuracy  
        # Convert None to string for plotting
        max_depth_plot = results_df['max_depth'].astype(str)
        unique_depths = sorted(max_depth_plot.unique())
        depth_means = [results_df[max_depth_plot == d]['cv_mean'].mean() for d in unique_depths]
        
        axes[0, 1].bar(unique_depths, depth_means, alpha=0.7)
        axes[0, 1].set_xlabel('max_depth')
        axes[0, 1].set_ylabel('Mean CV Accuracy')
        axes[0, 1].set_title('max_depth vs Performance')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. min_samples_split vs accuracy
        axes[1, 0].scatter(results_df['min_samples_split'], results_df['cv_mean'], alpha=0.7)
        axes[1, 0].set_xlabel('min_samples_split')
        axes[1, 0].set_ylabel('CV Accuracy')
        axes[1, 0].set_title('min_samples_split vs Performance')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Heatmap de top combinations
        # Get top 10 combinations
        top_results = results_df.nlargest(10, 'cv_mean')
        
        # Create parameter combination strings
        param_combos = []
        for _, row in top_results.iterrows():
            combo = f"est:{row['n_estimators']}, depth:{row['max_depth']}"
            param_combos.append(combo)
        
        y_pos = np.arange(len(param_combos))
        axes[1, 1].barh(y_pos, top_results['cv_mean'], alpha=0.7)
        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels(param_combos, fontsize=8)
        axes[1, 1].set_xlabel('CV Accuracy')
        axes[1, 1].set_title('Top 10 Parameter Combinations')
        
        plt.tight_layout()
        plt.savefig("parameter_importance_analysis.png", dpi=300, bbox_inches='tight')
        mlflow.log_artifact("parameter_importance_analysis.png", "optimization_analysis")
        plt.close()

def demo_experiment_organization():
    """Demo completo de organización de experimentos."""
    
    print("🎯 Advanced Experiment Organization Demo")
    print("=" * 60)
    
    organizer = ExperimentOrganizer("iris-classification")
    
    # 1. Crear jerarquía de experimentos
    print("\n1️⃣ Creating experiment hierarchy...")
    organizer.create_experiment_hierarchy()
    
    # 2. Ejecutar estudio baseline
    print("\n2️⃣ Running baseline study...")
    baseline_results, baseline_run_id = organizer.baseline_study()
    
    # 3. Ejecutar optimización de hiperparámetros
    print("\n3️⃣ Running hyperparameter optimization...")
    best_params, best_score, opt_run_id = organizer.hyperparameter_optimization_study()
    
    print(f"""
    🎉 Advanced tracking demo completed!
    
    📁 Experiments organized hierarchically
    🏆 Best baseline: {max(baseline_results, key=lambda x: baseline_results[x]['test_accuracy'])}
    🎯 Optimized params: {best_params}
    📈 Best score: {best_score:.4f}
    
    🔗 Explore results: {config.MLFLOW_TRACKING_URI}
    """)
    
    return baseline_results, best_params

if __name__ == "__main__":
    baseline_results, best_params = demo_experiment_organization()
```

**Ejecutar organización avanzada:**
```bash
uv run python -m src.experiments.advanced_organization
```

### Lab 2: Auto-logging Mastery (25 minutos)

#### Crear src/experiments/autolog_mastery.py:
```python
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Agregar src al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.config import config
from src.data.load_data import load_iris_data, get_train_test_split

class AutologMaster:
    """Clase para demostrar auto-logging avanzado."""
    
    def __init__(self):
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment("iris-classification/04-autolog-mastery")
        
        # Preparar datos
        X, y = load_iris_data()
        self.X_train, self.X_test, self.y_train, self.y_test = get_train_test_split(X, y)
    
    def sklearn_autolog_demo(self):
        """Demo completo de scikit-learn auto-logging."""
        
        with mlflow.start_run(run_name="sklearn_autolog_comprehensive") as parent_run:
            # Configurar auto-logging con todas las opciones
            mlflow.sklearn.autolog(
                log_input_examples=True,      # Log ejemplos de input
                log_model_signatures=True,    # Log schema input/output
                log_models=True,              # Log modelos entrenados
                log_datasets=True,            # Log información de datasets
                disable=False,                # Habilitar auto-logging
                exclusive=False,              # Permitir logging manual también
                disable_for_unsupported_versions=False,
                silent=False,                 # Mostrar información de auto-logging
                registered_model_name=None,   # No registrar automáticamente
                pos_label=None,               # Para métricas binarias
                extra_tags=None               # Tags adicionales
            )
            
            # Tags para el estudio
            mlflow.set_tags({
                "autolog_framework": "scikit-learn",
                "autolog_features": "comprehensive",
                "demo_type": "autolog_showcase"
            })
            
            models_to_test = {
                "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
                "logistic_regression": LogisticRegression(random_state=42, max_iter=1000),
                "svm": SVC(kernel='rbf', random_state=42, probability=True)  # probability=True para predict_proba
            }
            
            autolog_results = {}
            
            for model_name, model in models_to_test.items():
                with mlflow.start_run(run_name=f"autolog_{model_name}", nested=True):
                    print(f"🤖 Auto-logging {model_name}...")
                    
                    # El auto-logging capturará automáticamente:
                    # - Todos los parámetros del modelo
                    # - Métricas de training (score)
                    # - El modelo entrenado
                    # - Ejemplos de input
                    # - Model signature
                    model.fit(self.X_train, self.y_train)
                    
                    # Podemos agregar métricas personalizadas encima del auto-logging
                    test_accuracy = model.score(self.X_test, self.y_test)
                    train_accuracy = model.score(self.X_train, self.y_train)
                    
                    # Custom business metrics
                    mlflow.log_metrics({
                        "custom_test_accuracy": test_accuracy,
                        "custom_train_accuracy": train_accuracy,
                        "custom_overfitting_ratio": train_accuracy / test_accuracy,
                        "custom_generalization_gap": train_accuracy - test_accuracy
                    })
                    
                    # Custom tags
                    mlflow.set_tags({
                        "model_complexity": "high" if model_name in ["random_forest", "gradient_boosting"] else "medium",
                        "interpretability": "low" if model_name in ["random_forest", "svm"] else "high"
                    })
                    
                    # Predicciones para análisis personalizado
                    if hasattr(model, 'predict_proba'):
                        y_proba = model.predict_proba(self.X_test)
                        confidence_scores = np.max(y_proba, axis=1)
                        
                        mlflow.log_metrics({
                            "mean_prediction_confidence": confidence_scores.mean(),
                            "min_prediction_confidence": confidence_scores.min(),
                            "std_prediction_confidence": confidence_scores.std()
                        })
                    
                    autolog_results[model_name] = {
                        "test_accuracy": test_accuracy,
                        "train_accuracy": train_accuracy
                    }
                    
                    print(f"   ✅ {model_name}: Test accuracy = {test_accuracy:.4f}")
            
            # Análisis comparativo en parent run
            best_model = max(autolog_results, key=lambda x: autolog_results[x]["test_accuracy"])
            mlflow.log_metrics({
                "best_autolog_model_accuracy": autolog_results[best_model]["test_accuracy"],
                "autolog_models_tested": len(autolog_results)
            })
            
            mlflow.log_params({"best_autolog_model": best_model})
            
            # Desactivar auto-logging después del demo
            mlflow.sklearn.autolog(disable=True)
            
            print(f"""
            🤖 Auto-logging demo completed!
            🏆 Best model: {best_model}
            📊 Best accuracy: {autolog_results[best_model]['test_accuracy']:.4f}
            """)
            
            return autolog_results
    
    def pipeline_autolog_demo(self):
        """Demo de auto-logging con pipelines de scikit-learn."""
        
        with mlflow.start_run(run_name="pipeline_autolog_demo"):
            # Auto-logging captura pipelines completos
            mlflow.sklearn.autolog(
                log_input_examples=True,
                log_model_signatures=True,
                log_models=True
            )
            
            mlflow.set_tags({
                "demo_type": "pipeline_autolog",
                "pipeline_steps": "scaling_and_classification"
            })
            
            # Crear pipeline con preprocessing y modelo
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
            
            print("🔧 Training pipeline with auto-logging...")
            
            # Auto-logging capturará:
            # - Parámetros de todos los pasos del pipeline
            # - El pipeline completo como modelo
            # - Métricas de evaluación
            pipeline.fit(self.X_train, self.y_train)
            
            # Métricas adicionales
            test_score = pipeline.score(self.X_test, self.y_test)
            
            mlflow.log_metrics({
                "pipeline_test_accuracy": test_score,
                "pipeline_steps_count": len(pipeline.steps)
            })
            
            # Información de cada paso
            for i, (step_name, step_obj) in enumerate(pipeline.steps):
                mlflow.log_params({
                    f"step_{i}_name": step_name,
                    f"step_{i}_type": type(step_obj).__name__
                })
            
            mlflow.sklearn.autolog(disable=True)
            
            print(f"🔧 Pipeline auto-logging completed! Accuracy: {test_score:.4f}")
            
            return test_score
    
    def gridsearch_autolog_demo(self):
        """Demo de auto-logging con GridSearchCV."""
        
        with mlflow.start_run(run_name="gridsearch_autolog_demo"):
            # Auto-logging para GridSearch es especialmente potente
            mlflow.sklearn.autolog(
                log_input_examples=True,
                log_model_signatures=True,
                log_models=True,
                log_post_training_metrics=True
            )
            
            mlflow.set_tags({
                "demo_type": "gridsearch_autolog",
                "search_method": "grid_search_cv"
            })
            
            # Definir grid de parámetros
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [3, 5, None],
                'min_samples_split': [2, 5]
            }
            
            # GridSearchCV con auto-logging
            grid_search = GridSearchCV(
                estimator=RandomForestClassifier(random_state=42),
                param_grid=param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            print("🔍 Running GridSearch with auto-logging...")
            
            # Auto-logging capturará automáticamente:
            # - Todos los parámetros del grid
            # - El mejor estimador
            # - Métricas de CV
            # - Resultados completos del grid search
            grid_search.fit(self.X_train, self.y_train)
            
            # Métricas del mejor modelo
            best_score = grid_search.best_score_
            test_score = grid_search.score(self.X_test, self.y_test)
            
            # Log información adicional del grid search
            mlflow.log_metrics({
                "gridsearch_best_cv_score": best_score,
                "gridsearch_test_score": test_score,
                "gridsearch_total_fits": len(param_grid['n_estimators']) * 
                                       len(param_grid['max_depth']) * 
                                       len(param_grid['min_samples_split']) * 5  # CV folds
            })
            
            # Best parameters como tags
            for param_name, param_value in grid_search.best_params_.items():
                mlflow.set_tag(f"best_{param_name}", str(param_value))
            
            mlflow.sklearn.autolog(disable=True)
            
            print(f"""
            🔍 GridSearch auto-logging completed!
            🏆 Best CV score: {best_score:.4f}
            🎯 Test score: {test_score:.4f}
            📊 Best params: {grid_search.best_params_}
            """)
            
            return grid_search.best_params_, best_score
    
    def xgboost_autolog_demo(self):
        """Demo de auto-logging con XGBoost."""
        
        with mlflow.start_run(run_name="xgboost_autolog_demo"):
            # Auto-logging para XGBoost
            mlflow.xgboost.autolog(
                log_input_examples=True,
                log_model_signatures=True,
                log_models=True,
                log_datasets=True,
                importance_types=["weight", "gain", "cover"]  # Tipos de feature importance
            )
            
            mlflow.set_tags({
                "demo_type": "xgboost_autolog",
                "framework": "xgboost"
            })
            
            # Preparar datos para XGBoost
            dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
            dtest = xgb.DMatrix(self.X_test, label=self.y_test)
            
            # Parámetros XGBoost
            params = {
                'objective': 'multi:softprob',
                'num_class': 3,
                'max_depth': 6,
                'eta': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'seed': 42
            }
            
            print("🚀 Training XGBoost with auto-logging...")
            
            # Auto-logging capturará:
            # - Parámetros de XGBoost
            # - Métricas de training por época
            # - Feature importance
            # - El modelo booster
            model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=100,
                evals=[(dtrain, 'train'), (dtest, 'eval')],
                early_stopping_rounds=10,
                verbose_eval=20
            )
            
            # Métricas adicionales
            train_preds = model.predict(dtrain)
            test_preds = model.predict(dtest)
            
            # Convert probabilities to class predictions
            train_pred_classes = np.argmax(train_preds, axis=1)
            test_pred_classes = np.argmax(test_preds, axis=1)
            
            train_accuracy = (train_pred_classes == self.y_train).mean()
            test_accuracy = (test_pred_classes == self.y_test).mean()
            
            mlflow.log_metrics({
                "xgb_train_accuracy": train_accuracy,
                "xgb_test_accuracy": test_accuracy,
                "xgb_num_boost_rounds": model.num_boosted_rounds()
            })
            
            mlflow.xgboost.autolog(disable=True)
            
            print(f"""
            🚀 XGBoost auto-logging completed!
            📈 Train accuracy: {train_accuracy:.4f}
            🎯 Test accuracy: {test_accuracy:.4f}
            🌳 Boosting rounds: {model.num_boosted_rounds()}
            """)
            
            return test_accuracy

def demo_autolog_mastery():
    """Demo completo de auto-logging mastery."""
    
    print("🤖 Auto-logging Mastery Demo")
    print("=" * 50)
    
    autolog_master = AutologMaster()
    
    # 1. Scikit-learn auto-logging comprehensivo
    print("\n1️⃣ Scikit-learn auto-logging demo...")
    sklearn_results = autolog_master.sklearn_autolog_demo()
    
    # 2. Pipeline auto-logging
    print("\n2️⃣ Pipeline auto-logging demo...")
    pipeline_score = autolog_master.pipeline_autolog_demo()
    
    # 3. GridSearch auto-logging
    print("\n3️⃣ GridSearch auto-logging demo...")
    best_params, best_score = autolog_master.gridsearch_autolog_demo()
    
    # 4. XGBoost auto-logging
    print("\n4️⃣ XGBoost auto-logging demo...")
    xgb_score = autolog_master.xgboost_autolog_demo()
    
    print(f"""
    🤖 Auto-logging mastery demo completed!
    
    📊 Results summary:
    - Best sklearn model: {max(sklearn_results, key=lambda x: sklearn_results[x]['test_accuracy'])}
    - Pipeline accuracy: {pipeline_score:.4f}
    - GridSearch best: {best_score:.4f}
    - XGBoost accuracy: {xgb_score:.4f}
    
    🔗 Explore auto-logged experiments: {config.MLFLOW_TRACKING_URI}
    """)

if __name__ == "__main__":
    demo_autolog_mastery()
```

**Ejecutar auto-logging mastery:**
```bash
# Instalar XGBoost
uv add xgboost --group ml

# Ejecutar demo
uv run python -m src.experiments.autolog_mastery
```

### Lab 3: Custom Artifacts & Advanced Analysis (25 minutos)

#### Crear src/experiments/custom_artifacts.py:
```python
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import learning_curve, validation_curve
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
import shap
import json
from datetime import datetime
import sys
import os

# Agregar src al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.config import config
from src.data.load_data import load_iris_data, get_train_test_split

class CustomArtifactsCreator:
    """Clase para crear artefactos personalizados ricos."""
    
    def __init__(self):
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment("iris-classification/05-custom-artifacts")
        
        # Preparar datos
        X, y = load_iris_data()
        self.X_train, self.X_test, self.y_train, self.y_test = get_train_test_split(X, y)
        
        # Feature names para análisis
        self.feature_names = [
            'sepal_length', 'sepal_width', 
            'petal_length', 'petal_width'
        ]
        
        # Class names
        self.class_names = ['setosa', 'versicolor', 'virginica']
    
    def create_interactive_plots(self, model, run_name="interactive_plots_demo"):
        """Crear plots interactivos con Plotly."""
        
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({
                "artifact_type": "interactive_plots",
                "visualization_library": "plotly"
            })
            
            # Entrenar modelo
            model.fit(self.X_train, self.y_train)
            
            # Predicciones
            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test)
            
            # 1. Interactive Feature Importance
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=True)
                
                fig_importance = px.bar(
                    importance_df, 
                    x='importance', 
                    y='feature',
                    orientation='h',
                    title='Interactive Feature Importance',
                    labels={'importance': 'Importance Score', 'feature': 'Features'},
                    color='importance',
                    color_continuous_scale='viridis'
                )
                
                fig_importance.update_layout(
                    height=500,
                    showlegend=False,
                    title_x=0.5
                )
                
                fig_importance.write_html("interactive_feature_importance.html")
                mlflow.log_artifact("interactive_feature_importance.html", "interactive_plots")
            
            # 2. Interactive Confusion Matrix
            cm = confusion_matrix(self.y_test, y_pred)
            
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=self.class_names,
                y=self.class_names,
                color_continuous_scale='Blues',
                title="Interactive Confusion Matrix"
            )
            
            # Add text annotations
            for i in range(len(self.class_names)):
                for j in range(len(self.class_names)):
                    fig_cm.add_annotation(
                        x=j, y=i,
                        text=str(cm[i, j]),
                        showarrow=False,
                        font=dict(color="white" if cm[i, j] > cm.max()/2 else "black")
                    )
            
            fig_cm.write_html("interactive_confusion_matrix.html")
            mlflow.log_artifact("interactive_confusion_matrix.html", "interactive_plots")
            
            # 3. Interactive Prediction Confidence
            confidence_scores = np.max(y_proba, axis=1)
            predicted_classes = [self.class_names[i] for i in y_pred]
            actual_classes = [self.class_names[i] for i in self.y_test]
            
            confidence_df = pd.DataFrame({
                'sample_id': range(len(self.y_test)),
                'predicted_class': predicted_classes,
                'actual_class': actual_classes,
                'confidence': confidence_scores,
                'correct': y_pred == self.y_test
            })
            
            fig_confidence = px.scatter(
                confidence_df,
                x='sample_id',
                y='confidence',
                color='correct',
                symbol='predicted_class',
                title='Prediction Confidence Analysis',
                labels={'sample_id': 'Sample ID', 'confidence': 'Prediction Confidence'},
                hover_data=['actual_class', 'predicted_class']
            )
            
            fig_confidence.update_layout(height=500, title_x=0.5)
            fig_confidence.write_html("interactive_confidence_analysis.html")
            mlflow.log_artifact("interactive_confidence_analysis.html", "interactive_plots")
            
            # 4. Interactive Learning Curves
            train_sizes, train_scores, val_scores = learning_curve(
                model, self.X_train, self.y_train, cv=5, n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10)
            )
            
            learning_curve_df = pd.DataFrame({
                'train_size': np.tile(train_sizes, 2),
                'score': np.concatenate([train_scores.mean(axis=1), val_scores.mean(axis=1)]),
                'std': np.concatenate([train_scores.std(axis=1), val_scores.std(axis=1)]),
                'type': ['Training'] * len(train_sizes) + ['Validation'] * len(train_sizes)
            })
            
            fig_learning = px.line(
                learning_curve_df,
                x='train_size',
                y='score',
                color='type',
                error_y='std',
                title='Interactive Learning Curves',
                labels={'train_size': 'Training Set Size', 'score': 'Accuracy Score'}
            )
            
            fig_learning.update_layout(height=500, title_x=0.5)
            fig_learning.write_html("interactive_learning_curves.html")
            mlflow.log_artifact("interactive_learning_curves.html", "interactive_plots")
            
            # Log métricas básicas
            test_accuracy = model.score(self.X_test, self.y_test)
            mlflow.log_metrics({
                "test_accuracy": test_accuracy,
                "mean_confidence": confidence_scores.mean(),
                "interactive_plots_created": 4
            })
            
            print(f"📊 Created 4 interactive plots with test accuracy: {test_accuracy:.4f}")
            
            return test_accuracy
    
    def create_model_explanations(self, model, run_name="model_explanations_demo"):
        """Crear explicaciones del modelo con SHAP."""
        
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({
                "artifact_type": "model_explanations",
                "explainability_library": "shap"
            })
            
            # Entrenar modelo
            model.fit(self.X_train, self.y_train)
            
            # SHAP Explainer
            explainer = shap.Explainer(model, self.X_train)
            shap_values = explainer(self.X_test)
            
            # 1. SHAP Summary Plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, self.X_test, feature_names=self.feature_names, show=False)
            plt.title('SHAP Summary Plot - Feature Importance')
            plt.tight_layout()
            plt.savefig("shap_summary_plot.png", dpi=300, bbox_inches='tight')
            mlflow.log_artifact("shap_summary_plot.png", "explanations")
            plt.close()
            
            # 2. SHAP Waterfall Plot para primera predicción
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(shap_values[0])
            plt.title('SHAP Waterfall Plot - First Prediction')
            plt.tight_layout()
            plt.savefig("shap_waterfall_plot.png", dpi=300, bbox_inches='tight')
            mlflow.log_artifact("shap_waterfall_plot.png", "explanations")
            plt.close()
            
            # 3. SHAP Beeswarm Plot
            plt.figure(figsize=(10, 8))
            shap.plots.beeswarm(shap_values, show=False)
            plt.title('SHAP Beeswarm Plot - Feature Impact Distribution')
            plt.tight_layout()
            plt.savefig("shap_beeswarm_plot.png", dpi=300, bbox_inches='tight')
            mlflow.log_artifact("shap_beeswarm_plot.png", "explanations")
            plt.close()
            
            # 4. Feature Importance Summary
            feature_importance = np.abs(shap_values.values).mean(0)
            importance_summary = pd.DataFrame({
                'feature': self.feature_names,
                'mean_shap_value': feature_importance.mean(axis=1) if len(feature_importance.shape) > 1 else feature_importance,
                'rank': range(1, len(self.feature_names) + 1)
            }).sort_values('mean_shap_value', ascending=False)
            
            importance_summary.to_csv("feature_importance_summary.csv", index=False)
            mlflow.log_artifact("feature_importance_summary.csv", "explanations")
            
            # 5. HTML Report con explicaciones
            html_report = self._create_explanation_html_report(model, shap_values, importance_summary)
            
            with open("model_explanation_report.html", "w") as f:
                f.write(html_report)
            mlflow.log_artifact("model_explanation_report.html", "reports")
            
            # Log métricas de explicabilidad
            test_accuracy = model.score(self.X_test, self.y_test)
            mlflow.log_metrics({
                "test_accuracy": test_accuracy,
                "most_important_feature_shap": importance_summary.iloc[0]['mean_shap_value'],
                "feature_importance_range": importance_summary['mean_shap_value'].max() - 
                                         importance_summary['mean_shap_value'].min(),
                "explanation_artifacts_created": 5
            })
            
            print(f"🔍 Created model explanations with test accuracy: {test_accuracy:.4f}")
            print(f"📊 Most important feature: {importance_summary.iloc[0]['feature']}")
            
            return test_accuracy, importance_summary
    
    def create_comprehensive_analysis_report(self, model, run_name="comprehensive_analysis"):
        """Crear reporte de análisis comprehensivo."""
        
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({
                "artifact_type": "comprehensive_analysis",
                "report_type": "full_model_analysis"
            })
            
            # Entrenar modelo
            model.fit(self.X_train, self.y_train)
            
            # Recopilar todos los análisis
            analysis_results = {}
            
            # 1. Métricas de rendimiento
            train_accuracy = model.score(self.X_train, self.y_train)
            test_accuracy = model.score(self.X_test, self.y_test)
            y_pred = model.predict(self.X_test)
            
            # Classification report
            class_report = classification_report(self.y_test, y_pred, output_dict=True)
            
            analysis_results['performance'] = {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'overfitting_ratio': train_accuracy / test_accuracy,
                'classification_report': class_report
            }
            
            # 2. Análisis de validación
            param_name = 'n_estimators'
            param_range = [10, 50, 100, 150, 200]
            
            train_scores, val_scores = validation_curve(
                RandomForestClassifier(random_state=42), 
                self.X_train, self.y_train,
                param_name=param_name, param_range=param_range,
                cv=5, scoring='accuracy'
            )
            
            analysis_results['validation'] = {
                'param_name': param_name,
                'param_range': param_range,
                'train_scores_mean': train_scores.mean(axis=1).tolist(),
                'val_scores_mean': val_scores.mean(axis=1).tolist(),
                'optimal_param': param_range[np.argmax(val_scores.mean(axis=1))]
            }
            
            # 3. Feature Analysis
            if hasattr(model, 'feature_importances_'):
                feature_analysis = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': model.feature_importances_,
                    'rank': range(1, len(self.feature_names) + 1)
                }).sort_values('importance', ascending=False)
                
                analysis_results['features'] = {
                    'most_important': feature_analysis.iloc[0]['feature'],
                    'least_important': feature_analysis.iloc[-1]['feature'],
                    'importance_distribution': feature_analysis['importance'].tolist()
                }
            
            # 4. Prediction Analysis
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(self.X_test)
                confidence_scores = np.max(y_proba, axis=1)
                
                analysis_results['predictions'] = {
                    'mean_confidence': confidence_scores.mean(),
                    'min_confidence': confidence_scores.min(),
                    'max_confidence': confidence_scores.max(),
                    'low_confidence_samples': (confidence_scores < 0.7).sum(),
                    'high_confidence_samples': (confidence_scores > 0.9).sum()
                }
            
            # 5. Crear reporte JSON
            analysis_results['metadata'] = {
                'model_type': type(model).__name__,
                'feature_count': len(self.feature_names),
                'class_count': len(self.class_names),
                'train_samples': len(self.X_train),
                'test_samples': len(self.X_test),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Guardar análisis como JSON
            with open("comprehensive_analysis.json", "w") as f:
                json.dump(analysis_results, f, indent=2, default=str)
            mlflow.log_artifact("comprehensive_analysis.json", "analysis")
            
            # Crear reporte HTML visual
            html_report = self._create_comprehensive_html_report(analysis_results)
            
            with open("comprehensive_analysis_report.html", "w") as f:
                f.write(html_report)
            mlflow.log_artifact("comprehensive_analysis_report.html", "reports")
            
            # Log métricas del análisis
            mlflow.log_metrics({
                "test_accuracy": test_accuracy,
                "analysis_components": 5,
                "optimal_n_estimators": analysis_results['validation']['optimal_param']
            })
            
            # Log parámetros del análisis
            mlflow.log_params({
                "analysis_type": "comprehensive",
                "validation_param": param_name,
                "feature_importance_available": hasattr(model, 'feature_importances_')
            })
            
            print(f"""
            📈 Comprehensive analysis completed!
            🎯 Test accuracy: {test_accuracy:.4f}
            🔍 Most important feature: {analysis_results['features']['most_important']}
            ⚙️ Optimal n_estimators: {analysis_results['validation']['optimal_param']}
            """)
            
            return analysis_results
    
    def _create_explanation_html_report(self, model, shap_values, importance_summary):
        """Crear reporte HTML con explicaciones del modelo."""
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Explanation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; border-left: 4px solid #3498db; padding-left: 15px; }}
                .metric {{ background-color: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .feature-importance {{ background-color: #e8f8f5; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .summary {{ background-color: #d5e8d4; padding: 20px; border-radius: 10px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔍 Model Explanation Report</h1>
                
                <div class="summary">
                    <h2>📊 Model Performance Summary</h2>
                    <div class="metric">
                        <strong>Model Type:</strong> {type(model).__name__}<br>
                        <strong>Test Accuracy:</strong> {model.score(self.X_test, self.y_test):.4f}<br>
                        <strong>Features:</strong> {len(self.feature_names)}<br>
                        <strong>Classes:</strong> {len(self.class_names)}
                    </div>
                </div>
                
                <h2>🎯 Feature Importance Analysis</h2>
                <div class="feature-importance">
                    <p>Features ranked by their impact on model predictions (SHAP values):</p>
                    <table>
                        <tr>
                            <th>Rank</th>
                            <th>Feature</th>
                            <th>Mean SHAP Value</th>
                            <th>Relative Importance</th>
                        </tr>
        """
        
        max_importance = importance_summary['mean_shap_value'].max()
        for _, row in importance_summary.iterrows():
            relative_importance = (row['mean_shap_value'] / max_importance) * 100
            html_template += f"""
                        <tr>
                            <td>{row['rank']}</td>
                            <td>{row['feature']}</td>
                            <td>{row['mean_shap_value']:.4f}</td>
                            <td>{relative_importance:.1f}%</td>
                        </tr>
            """
        
        html_template += """
                    </table>
                </div>
                
                <h2>💡 Key Insights</h2>
                <div class="summary">
                    <ul>
        """
        
        # Agregar insights automáticos
        top_feature = importance_summary.iloc[0]['feature']
        top_importance = importance_summary.iloc[0]['mean_shap_value']
        
        html_template += f"""
                        <li><strong>Most Important Feature:</strong> {top_feature} contributes most to predictions</li>
                        <li><strong>Feature Dominance:</strong> Top feature has {top_importance:.4f} average impact</li>
                        <li><strong>Model Interpretability:</strong> Feature importance is well-distributed across features</li>
                    </ul>
                </div>
                
                <h2>📈 Visualizations</h2>
                <p>Check the artifacts section for interactive SHAP plots including:</p>
                <ul>
                    <li>Summary Plot - Overall feature importance</li>
                    <li>Waterfall Plot - Individual prediction breakdown</li>
                    <li>Beeswarm Plot - Feature impact distribution</li>
                </ul>
                
                <div style="text-align: center; margin-top: 30px; color: #7f8c8d;">
                    <p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def _create_comprehensive_html_report(self, analysis_results):
        """Crear reporte HTML comprehensivo."""
        
        performance = analysis_results['performance']
        validation = analysis_results['validation']
        features = analysis_results['features']
        predictions = analysis_results['predictions']
        metadata = analysis_results['metadata']
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Comprehensive Model Analysis Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background-color: #f8f9fa; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; }}
                .container {{ max-width: 1200px; margin: 0 auto; padding: 30px; }}
                .section {{ background-color: white; margin: 20px 0; padding: 25px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
                .metric-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }}
                .metric-value {{ font-size: 2em; font-weight: bold; }}
                .metric-label {{ font-size: 0.9em; opacity: 0.9; }}
                h1 {{ margin: 0; font-size: 2.5em; }}
                h2 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                .insight {{ background-color: #e8f4fd; border-left: 4px solid #3498db; padding: 15px; margin: 10px 0; }}
                .warning {{ background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 10px 0; }}
                .success {{ background-color: #d4edda; border-left: 4px solid #28a745; padding: 15px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Comprehensive Model Analysis</h1>
                <p>Complete performance and behavior analysis for {metadata['model_type']}</p>
            </div>
            
            <div class="container">
                <div class="section">
                    <h2>🎯 Performance Overview</h2>
                    <div class="metric-grid">
                        <div class="metric-card">
                            <div class="metric-value">{performance['test_accuracy']:.3f}</div>
                            <div class="metric-label">Test Accuracy</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{performance['train_accuracy']:.3f}</div>
                            <div class="metric-label">Train Accuracy</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{performance['overfitting_ratio']:.3f}</div>
                            <div class="metric-label">Overfitting Ratio</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{metadata['feature_count']}</div>
                            <div class="metric-label">Features</div>
                        </div>
                    </div>
                    
                    <div class="{'success' if performance['overfitting_ratio'] <= 1.1 else 'warning'}">
                        <strong>Overfitting Analysis:</strong> 
                        {'✅ Good generalization - minimal overfitting detected' if performance['overfitting_ratio'] <= 1.1 else '⚠️ Potential overfitting - consider regularization'}
                    </div>
                </div>
                
                <div class="section">
                    <h2>🔧 Hyperparameter Optimization</h2>
                    <div class="insight">
                        <strong>Optimal {validation['param_name']}:</strong> {validation['optimal_param']}<br>
                        <strong>Parameter Range Tested:</strong> {validation['param_range']}<br>
                        <strong>Best Validation Score:</strong> {max(validation['val_scores_mean']):.4f}
                    </div>
                </div>
                
                <div class="section">
                    <h2>🌟 Feature Importance</h2>
                    <div class="insight">
                        <strong>Most Important Feature:</strong> {features['most_important']}<br>
                        <strong>Least Important Feature:</strong> {features['least_important']}<br>
                        <strong>Feature Distribution:</strong> Well-balanced importance across features
                    </div>
                </div>
                
                <div class="section">
                    <h2>🎲 Prediction Analysis</h2>
                    <div class="metric-grid">
                        <div class="metric-card">
                            <div class="metric-value">{predictions['mean_confidence']:.3f}</div>
                            <div class="metric-label">Mean Confidence</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{predictions['high_confidence_samples']}</div>
                            <div class="metric-label">High Confidence</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{predictions['low_confidence_samples']}</div>
                            <div class="metric-label">Low Confidence</div>
                        </div>
                    </div>
                    
                    <div class="{'success' if predictions['mean_confidence'] > 0.8 else 'warning'}">
                        <strong>Confidence Analysis:</strong> 
                        {'✅ High prediction confidence indicates reliable model' if predictions['mean_confidence'] > 0.8 else '⚠️ Lower confidence may indicate need for more training data'}
                    </div>
                </div>
                
                <div class="section">
                    <h2>💡 Key Recommendations</h2>
                    <div class="insight">
                        <h3>Model Strengths:</h3>
                        <ul>
                            <li>Achieves {performance['test_accuracy']:.1%} accuracy on test data</li>
                            <li>{'Good generalization with minimal overfitting' if performance['overfitting_ratio'] <= 1.1 else 'Shows some overfitting - consider regularization'}</li>
                            <li>{'High prediction confidence' if predictions['mean_confidence'] > 0.8 else 'Moderate prediction confidence'}</li>
                        </ul>
                        
                        <h3>Next Steps:</h3>
                        <ul>
                            <li>Consider feature engineering on {features['least_important']} to improve performance</li>
                            <li>Experiment with {validation['param_name']} around optimal value of {validation['optimal_param']}</li>
                            <li>{'Monitor for data drift in production' if performance['test_accuracy'] > 0.9 else 'Consider collecting more training data'}</li>
                        </ul>
                    </div>
                </div>
                
                <div style="text-align: center; margin-top: 30px; color: #6c757d;">
                    <p>Analysis completed on {metadata['analysis_timestamp']}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_template

def demo_custom_artifacts():
    """Demo completo de artefactos personalizados."""
    
    print("🎨 Custom Artifacts Creation Demo")
    print("=" * 50)
    
    creator = CustomArtifactsCreator()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # 1. Interactive plots
    print("\n1️⃣ Creating interactive plots...")
    interactive_accuracy = creator.create_interactive_plots(model, "interactive_plots_showcase")
    
    # 2. Model explanations
    print("\n2️⃣ Creating model explanations...")
    explanation_accuracy, importance_summary = creator.create_model_explanations(
        RandomForestClassifier(n_estimators=100, random_state=42), 
        "model_explanations_showcase"
    )
    
    # 3. Comprehensive analysis
    print("\n3️⃣ Creating comprehensive analysis...")
    analysis_results = creator.create_comprehensive_analysis_report(
        RandomForestClassifier(n_estimators=100, random_state=42),
        "comprehensive_analysis_showcase"
    )
    
    print(f"""
    🎨 Custom artifacts demo completed!
    
    📊 Results:
    - Interactive plots accuracy: {interactive_accuracy:.4f}
    - Explanations accuracy: {explanation_accuracy:.4f}
    - Comprehensive analysis accuracy: {analysis_results['performance']['test_accuracy']:.4f}
    
    🏆 Most important feature: {importance_summary.iloc[0]['feature']}
    ⚙️ Optimal n_estimators: {analysis_results['validation']['optimal_param']}
    
    🔗 Explore rich artifacts: {config.MLFLOW_TRACKING_URI}
    """)

if __name__ == "__main__":
    # Instalar dependencias adicionales
    import subprocess
    try:
        import plotly
        import shap
    except ImportError:
        print("Installing required packages...")
        subprocess.run(["uv", "add", "plotly", "shap", "--group", "ml"])
        print("Please restart and run again.")
        exit()
    
    demo_custom_artifacts()
```

**Ejecutar custom artifacts:**
```bash
# Instalar dependencias adicionales
uv add plotly shap --group ml

# Ejecutar demo
uv run python -m src.experiments.custom_artifacts
```

---

## 🎯 Cierre de Sesión 2A (10 minutos)

### Resumen de Logros (5 minutos)
**Lo que dominamos hoy:**
1. ✅ Organización profesional de experimentos con naming conventions
2. ✅ Nested runs para estudios complejos (baseline, optimization)
3. ✅ Auto-logging mastery con scikit-learn, pipelines, GridSearch, XGBoost
4. ✅ Artefactos personalizados ricos (HTML interactivo, SHAP, análisis comprehensivo)
5. ✅ Análisis programático de experimentos y comparaciones

### Vista Previa Sesión 2B (3 minutos)
**Próxima sesión (2 horas):**
- Model Registry y gestión del ciclo de vida de modelos
- Promoción automatizada con validaciones
- Tool selection strategy (MLFlow vs DVC)
- Introducción a testing básico para ML

### Q&A y Troubleshooting (2 minutos)
**Problemas comunes:**
- Nested runs no aparecen → Verificar `nested=True` en start_run
- Auto-logging no funciona → Verificar versión de sklearn y `disable=False`
- Plots interactivos no se generan → Instalar plotly y shap

---

## 📋 Verificación de Aprendizaje

### Checkpoint Práctico:
- [ ] Experimentos organizados jerárquicamente creados
- [ ] Al menos 1 estudio con nested runs completado
- [ ] Auto-logging demo funcionando con múltiples frameworks
- [ ] Artefactos personalizados creados (HTML, plots, análisis)
- [ ] MLFlow UI mostrando todos los experimentos organizados

### Preparación para Sesión 2B:
- [ ] Todos los experimentos de 2A funcionando correctamente
- [ ] Al menos 10+ runs en MLFlow de diferentes tipos
- [ ] Entorno estable sin errores de dependencias
- [ ] Git repository actualizado con nuevo código

**🚀 ¡Listos para Model Registry y estrategias de herramientas en Sesión 2B!**