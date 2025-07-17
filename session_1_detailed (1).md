# Session 1: MLOps Fundamentals & Environment Setup
**Duration:** 4 hours  
**Format:** Theory (1.5h) + Hands-on Lab (2.5h)  
**Class Size:** 8-20 students  
**Prerequisites:** Basic Python, ML concepts, Git basics

---

## 🎯 Session Learning Objectives

By the end of this session, students will be able to:
- Explain what MLOps is and why it's critical for modern ML projects
- Identify the key challenges in ML lifecycle management
- Set up a professional MLOps development environment using modern tools
- Create their first MLFlow experiment with proper tracking
- Apply best practices for ML project organization

---

## 📚 Pre-Session Preparation

### Instructor Setup
- [ ] MLFlow demo environment running
- [ ] Sample datasets downloaded
- [ ] Code repository prepared
- [ ] Backup internet connection (for uv installation)
- [ ] Presentation slides ready

### Student Prerequisites
- [ ] Laptop with admin privileges
- [ ] Git installed and configured
- [ ] GitHub account created
- [ ] Basic terminal/command line familiarity
- [ ] Python 3.8+ installed (any version for now)

---

## 🕐 Part 1: MLOps Introduction (30 minutes)

### Opening Hook (5 minutes)
**Start with a real scenario:**
*"Imagine you're a data scientist at Spotify. You've built an amazing recommendation model that works perfectly on your laptop with 95% accuracy. But when you try to deploy it to production to serve 400 million users, everything breaks. The model predictions are different, the code crashes, and nobody can reproduce your results. This is why we need MLOps."*

### Core Content (20 minutes)

#### What is MLOps? (8 minutes)
**Definition:**
> MLOps is the practice of combining Machine Learning (ML) development with DevOps principles to automate and improve the continuous delivery of ML models to production.

**Key Components:**
1. **Model Development** - Experiment tracking, version control
2. **Model Deployment** - Automated pipelines, containerization
3. **Model Monitoring** - Performance tracking, drift detection
4. **Model Governance** - Compliance, security, approvals

**Visual Analogy:**
- Traditional Software: Code → Build → Test → Deploy → Monitor
- MLOps: Data + Code + Model → Experiment → Validate → Deploy → Monitor → Retrain

#### Why MLOps Matters (7 minutes)
**The ML Crisis Statistics:**
- 87% of ML projects never make it to production
- Average time to deploy: 6-12 months
- 90% of models degrade within 6 months without monitoring

**Real Problems MLOps Solves:**
1. **"It works on my machine"** → Reproducible environments
2. **"Where's version 2.3 of the model?"** → Model versioning
3. **"Is the model still accurate?"** → Automated monitoring
4. **"Who approved this model?"** → Governance workflows

**Business Impact:**
- Netflix: MLOps reduces model deployment time from months to days
- Uber: Automated retraining prevents 40% drop in model accuracy
- Airbnb: MLOps platform serves 150+ models to 1M+ predictions/second

#### MLOps vs Traditional DevOps (5 minutes)
| Aspect | Traditional DevOps | MLOps |
|--------|-------------------|-------|
| **Artifacts** | Code | Code + Data + Models |
| **Testing** | Unit tests | Data tests + Model tests |
| **Deployment** | Blue/green | A/B testing + Canary |
| **Monitoring** | System metrics | Model performance + drift |
| **Rollback** | Previous code version | Previous model + retraining |

### Interactive Discussion (5 minutes)
**Questions for students:**
1. "What ML projects have you worked on? What challenges did you face getting them into production?"
2. "In your current/past projects, how do you track experiments?"
3. "How do you know if your deployed model is still working correctly?"

---

## 🕐 Part 2: MLFlow Ecosystem Overview (45 minutes)

### MLFlow Introduction (10 minutes)
**What is MLFlow?**
MLFlow is an open-source platform for managing the complete machine learning lifecycle, including experimentation, reproducibility, deployment, and a central model registry.

**Why MLFlow?**
- **Open Source** - No vendor lock-in, active community
- **Language Agnostic** - Python, R, Java, Scala support
- **Framework Agnostic** - Works with scikit-learn, TensorFlow, PyTorch, etc.
- **Cloud Agnostic** - Runs on AWS, Azure, GCP, or on-premises

### MLFlow Components Deep Dive (25 minutes)

#### 1. MLFlow Tracking (8 minutes)
**Purpose:** Record and query experiments (code, data, config, results)

**Key Concepts:**
- **Experiment:** Named group of runs (e.g., "customer-churn-model")
- **Run:** Single execution of ML code
- **Parameters:** Input values (hyperparameters, config)
- **Metrics:** Numeric values to optimize (accuracy, RMSE)
- **Artifacts:** Output files (models, plots, data)

**Live Demo:** Show MLFlow UI with example experiment

#### 2. MLFlow Projects (7 minutes)
**Purpose:** Package ML code for reproducible runs

**Key Features:**
- **MLproject file:** Defines entry points and dependencies
- **Environment management:** Conda, Docker, virtualenv
- **Remote execution:** Run on different environments
- **Parameter passing:** Command-line interface

**Example MLproject file:**
```yaml
name: My ML Project
conda_env: conda.yaml
entry_points:
  main:
    parameters:
      alpha: {type: float, default: 0.5}
    command: "python train.py --alpha {alpha}"
```

#### 3. MLFlow Models (5 minutes)
**Purpose:** Standard format for packaging ML models

**Key Benefits:**
- **Multiple flavors:** scikit-learn, tensorflow, pytorch, etc.
- **Unified interface:** Same prediction API regardless of framework
- **Deployment ready:** Can be deployed to various platforms
- **Metadata:** Model signature, requirements, input/output schema

#### 4. MLFlow Model Registry (5 minutes)
**Purpose:** Central store for managing model lifecycle

**Key Features:**
- **Model versioning:** Track model evolution
- **Stage transitions:** Development → Staging → Production
- **Approval workflows:** Control model promotion
- **Lineage tracking:** Connect models to experiments

### MLFlow Architecture & Deployment (10 minutes)

#### Deployment Patterns
1. **Local Development:**
   ```bash
   mlflow ui  # Runs on localhost:5000
   ```

2. **Team Setup:**
   ```bash
   mlflow server --backend-store-uri sqlite:///mlflow.db \
                 --default-artifact-root ./artifacts \
                 --host 0.0.0.0 --port 5000
   ```

3. **Production Setup:**
   ```bash
   mlflow server --backend-store-uri postgresql://user:pass@host/db \
                 --default-artifact-root s3://bucket/path \
                 --host 0.0.0.0 --port 5000
   ```

#### Integration Ecosystem
- **Cloud Platforms:** AWS SageMaker, Azure ML, GCP Vertex AI
- **Orchestration:** Apache Airflow, Kubeflow, Prefect
- **Monitoring:** Prometheus, Grafana, DataDog
- **Storage:** S3, Azure Blob, GCS, HDFS

---

## 🕐 Part 3: Development Environment Best Practices (15 minutes)

### Modern Python Environment Management (8 minutes)
**Why Environment Management Matters:**
- Dependency conflicts between projects
- Reproducibility across team members
- Security (avoid global package installation)
- Version consistency

**Evolution of Python Tools:**
- **Old way:** `pip install` globally (💀 dependency hell)
- **Better:** `virtualenv` + `pip` (manual management)
- **Good:** `conda` (heavy but comprehensive)
- **Modern:** `uv` (fast, simple, modern)

**Why uv?**
- **10-100x faster** than pip
- **Rust-based** for speed and reliability
- **Simple syntax** easy to learn
- **Modern standards** (follows PEP standards)
- **Docker-friendly** for deployment

### Version Control for ML Projects (4 minutes)
**ML-Specific Git Challenges:**
- Large datasets can't go in Git
- Jupyter notebooks create messy diffs
- Model files are binary and large
- Experiment results need tracking

**Best Practices:**
- `.gitignore` for data/ and models/ directories
- Commit notebooks with cleared outputs
- Use MLFlow for experiment results, not Git
- Separate data/model versioning from code

### Reproducibility Principles (3 minutes)
**The Reproducibility Pyramid:**
1. **Environment** - Same packages, same versions
2. **Data** - Same datasets, same preprocessing
3. **Code** - Same algorithms, same parameters
4. **Infrastructure** - Same hardware, same OS

**Tools for Each Level:**
- Environment: `uv`, Docker
- Data: MLFlow artifacts, checksums
- Code: Git, MLFlow projects
- Infrastructure: Docker, cloud templates

---

## 🛠️ Part 4: Hands-On Lab (2.5 hours)

### Lab 1: Environment Setup with uv (45 minutes)

#### Step 1: Install uv (10 minutes)
**For students to follow along:**

```bash
# Install uv (fast Python package manager)
# macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows:
powershell -c "irm https://astral.sh/uv/install.sh | iex"

# Verify installation
uv --version
```

**Troubleshooting:**
- If curl fails: Download from https://github.com/astral-sh/uv/releases
- If Windows security blocks: Use `--execution-policy bypass`
- If behind corporate firewall: Use alternative installation method

#### Step 2: Create Project Structure (15 minutes)
```bash
# Create MLOps course project
uv init mlops-course
cd mlops-course

# Examine the generated structure
ls -la
cat pyproject.toml
```

**Explain pyproject.toml:**
```toml
[project]
name = "mlops-course"
version = "0.1.0"
description = "MLOps course using MLFlow"
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
dependencies = []

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

#### Step 3: Add Dependencies (10 minutes)
```bash
# Add core ML dependencies
uv add mlflow[extras] scikit-learn pandas numpy matplotlib seaborn

# Add API dependencies
uv add fastapi uvicorn python-dotenv

# Add development tools (separate group)
uv add pytest black ruff mypy pre-commit --group dev

# Add optional ML libraries
uv add optuna sentence-transformers --group ml

# Activate environment
uv sync
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows
```

**Verify installation:**
```bash
python -c "import mlflow; print(mlflow.__version__)"
mlflow --version
```

#### Step 4: Project Structure Setup (10 minutes)
```bash
# Create professional project structure
mkdir -p src/{data,models,evaluation,utils}
mkdir -p tests/{unit,integration}
mkdir -p notebooks
mkdir -p docker
mkdir -p .github/workflows

# Create __init__.py files
touch src/__init__.py
touch src/data/__init__.py
touch src/models/__init__.py
touch src/evaluation/__init__.py
touch src/utils/__init__.py
```

**Create .gitignore:**
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

# Virtual environments
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

# Data
data/raw/
data/processed/
*.csv
*.xlsx
*.json
*.pkl

# Models
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

### Lab 2: Project Setup & Git Integration (30 minutes)

#### Step 1: Git Repository Setup (10 minutes)
```bash
# Initialize Git repository
git init
git add .gitignore
git commit -m "Initial commit: project structure"

# Connect to GitHub (students should create repo first)
git remote add origin https://github.com/USERNAME/mlops-course.git
git branch -M main
git push -u origin main
```

#### Step 2: Environment Configuration (10 minutes)
**Create .env file:**
```bash
# Create environment configuration
cat > .env << EOF
# MLFlow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=iris-classification

# Database Configuration (for later use)
DATABASE_URL=sqlite:///mlflow.db

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Development Settings
DEBUG=True
LOG_LEVEL=INFO
EOF
```

**Create src/utils/config.py:**
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

#### Step 3: MLFlow Server Setup (10 minutes)
```bash
# Start MLFlow UI in background
mlflow ui --host 0.0.0.0 --port 5000 &

# Verify it's running
curl http://localhost:5000

# Open in browser
echo "MLFlow UI: http://localhost:5000"
```

**Show students the MLFlow UI:**
- Navigate to http://localhost:5000
- Explain the interface: Experiments, Runs, Models
- Show empty state before creating experiments

### Lab 3: First MLFlow Experiment (75 minutes)

#### Step 1: Create Sample Dataset (15 minutes)
**Create src/data/load_data.py:**
```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from typing import Tuple
import os

def load_iris_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load and return iris dataset as pandas objects."""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')
    return X, y

def load_wine_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load and return wine dataset as pandas objects."""
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target, name='target')
    return X, y

def get_train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """Split data into train and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def save_data(X_train, X_test, y_train, y_test, data_dir: str = "data"):
    """Save train/test splits to CSV files."""
    os.makedirs(data_dir, exist_ok=True)
    
    # Save training data
    train_data = X_train.copy()
    train_data['target'] = y_train
    train_data.to_csv(f"{data_dir}/train.csv", index=False)
    
    # Save test data
    test_data = X_test.copy()
    test_data['target'] = y_test
    test_data.to_csv(f"{data_dir}/test.csv", index=False)
    
    print(f"Data saved to {data_dir}/")
    print(f"Train shape: {train_data.shape}")
    print(f"Test shape: {test_data.shape}")

if __name__ == "__main__":
    # Load data
    X, y = load_iris_data()
    
    # Split data
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)
    
    # Save data
    save_data(X_train, X_test, y_train, y_test)
```

**Run data preparation:**
```bash
python src/data/load_data.py
ls data/
head data/train.csv
```

#### Step 2: Create Basic ML Pipeline (25 minutes)
**Create src/models/train.py:**
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
    """Load training and test data."""
    train_data = pd.read_csv("data/train.csv")
    test_data = pd.read_csv("data/test.csv")
    
    X_train = train_data.drop('target', axis=1)
    y_train = train_data['target']
    X_test = test_data.drop('target', axis=1)
    y_test = test_data['target']
    
    return X_train, X_test, y_train, y_test

def train_model(model_type: str = "random_forest", **model_params):
    """Train a model with MLFlow tracking."""
    
    # Set MLFlow tracking URI and experiment
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    
    with mlflow.start_run() as run:
        # Log run information
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("developer", "student")
        mlflow.set_tag("purpose", "training")
        
        # Load data
        X_train, X_test, y_train, y_test = load_data()
        
        # Log data information
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("features", X_train.shape[1])
        mlflow.log_param("classes", len(np.unique(y_train)))
        
        # Create model
        if model_type == "random_forest":
            model = RandomForestClassifier(**model_params)
        elif model_type == "logistic_regression":
            model = LogisticRegression(**model_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Log model parameters
        for param, value in model_params.items():
            mlflow.log_param(param, value)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        test_precision = precision_score(y_test, y_pred_test, average='weighted')
        test_recall = recall_score(y_test, y_pred_test, average='weighted')
        test_f1 = f1_score(y_test, y_pred_test, average='weighted')
        
        # Log metrics
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1", test_f1)
        
        # Create and save plots
        create_confusion_matrix_plot(y_test, y_pred_test)
        create_feature_importance_plot(model, X_train.columns)
        
        # Log model
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name=f"iris-{model_type}"
        )
        
        # Print results
        print(f"Run ID: {run.info.run_id}")
        print(f"Model: {model_type}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"MLFlow UI: {config.MLFLOW_TRACKING_URI}")
        
        return model, run.info.run_id

def create_confusion_matrix_plot(y_true, y_pred):
    """Create and save confusion matrix plot."""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png", "plots")
    plt.close()

def create_feature_importance_plot(model, feature_names):
    """Create and save feature importance plot."""
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.title('Feature Importance')
        plt.xlabel('Importance')
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
    print(f"Training completed! Check MLFlow UI: {config.MLFLOW_TRACKING_URI}")
```

#### Step 3: Run First Experiments (20 minutes)
```bash
# Run first experiment with default parameters
python src/models/train.py

# Run experiment with different parameters
python src/models/train.py --model-type random_forest --n-estimators 50 --max-depth 5

# Run logistic regression
python src/models/train.py --model-type logistic_regression

# Run multiple experiments with different parameters
python src/models/train.py --n-estimators 200 --max-depth 10
python src/models/train.py --n-estimators 50 --max-depth 3
```

#### Step 4: Explore MLFlow UI (15 minutes)
**Guide students through MLFlow UI:**

1. **Navigate to http://localhost:5000**
2. **Explore Experiments:**
   - Click on the experiment name
   - Compare different runs
   - Sort by metrics

3. **Examine Individual Runs:**
   - Click on a run ID
   - Review parameters, metrics, artifacts
   - Download artifacts

4. **Compare Runs:**
   - Select multiple runs
   - Click "Compare" button
   - Analyze parameter vs metric relationships

5. **Model Registry:**
   - Navigate to "Models" tab
   - See registered models
   - Explore model versions

**Discussion Questions:**
- Which model performed best and why?
- How do different parameters affect performance?
- What information is most useful for debugging?

---

## 🎯 Session Wrap-up (10 minutes)

### Key Takeaways Review (5 minutes)
**What we accomplished:**
1. ✅ Understood MLOps fundamentals and business value
2. ✅ Set up professional development environment with uv
3. ✅ Created first MLFlow experiments with proper tracking
4. ✅ Explored MLFlow UI for experiment comparison
5. ✅ Established project structure and best practices

### Preview Next Session (3 minutes)
**Session 2 Preview:**
- Advanced MLFlow Model Registry
- Strategic tool selection (MLFlow vs DVC decision framework)
- Comprehensive ML testing strategies
- Model promotion workflows

### Q&A and Troubleshooting (2 minutes)
**Common Issues:**
- MLFlow UI not loading → Check port 5000, restart server
- uv installation problems → Alternative installation methods
- Import errors → Check virtual environment activation

---

## 📋 Assessment Checkpoint

### Knowledge Check Questions:
1. What are the four main components of MLFlow?
2. When would you choose MLFlow artifacts over DVC for data versioning?
3. What's the difference between parameters and metrics in MLFlow?
4. How do you ensure reproducibility in ML experiments?

### Practical Verification:
- [ ] Student has working uv environment
- [ ] MLFlow UI accessible and showing experiments
- [ ] At least 3 experiment runs logged
- [ ] Project structure follows best practices
- [ ] Git repository created and committed

---

## 📎 Additional Resources

### For Students:
- [MLFlow Documentation](https://mlflow.org/docs/latest/)
- [uv Documentation](https://docs.astral.sh/uv/)
- [Git Best Practices for ML](https://dvc.org/doc/user-guide/how-to/data-management/git-best-practices)

### For Instructors:
- Common troubleshooting solutions
- Extended exercises for advanced students
- Alternative datasets for variety

---

## 🔧 Troubleshooting Guide

### Common Student Issues:

#### 1. uv Installation Fails
**Symptoms:** Command not found, SSL errors
**Solutions:**
- Use alternative installation method
- Check corporate firewall settings
- Download binary manually from GitHub releases

#### 2. MLFlow UI Won't Start
**Symptoms:** Port 5000 already in use, connection refused
**Solutions:**
```bash
# Check what's using port 5000
lsof -i :5000  # macOS/Linux
netstat -ano | findstr 5000  # Windows

# Use different port
mlflow ui --port 5001

# Kill existing process
kill -9 [PID]
```

#### 3. Import Errors
**Symptoms:** ModuleNotFoundError
**Solutions:**
```bash
# Verify virtual environment is activated
which python
echo $VIRTUAL_ENV

# Reinstall dependencies
uv sync --reinstall

# Check Python path
python -c "import sys; print(sys.path)"
```

#### 4. Git Repository Issues
**Symptoms:** Permission denied, authentication failed
**Solutions:**
- Set up SSH keys for GitHub
- Use personal access tokens
- Configure Git credentials properly

This detailed session plan provides everything needed to successfully teach Session 1, with clear timing, hands-on exercises, and troubleshooting guidance.