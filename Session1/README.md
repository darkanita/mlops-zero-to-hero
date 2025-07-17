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
    DEBUG = os.getenv("DEBUG", "False").lower(
