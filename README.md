# MLOps with MLFlow: Zero to Hero Course
**Duration:** 20 hours (5 sessions × 4 hours)  
**Format:** Theoretical + Practical  
**Tech Stack:** Python, MLFlow (Latest Version), Docker, Git

## 🎯 Course Overview

This comprehensive course teaches MLOps fundamentals using MLFlow as the primary platform. Students will learn to manage the complete ML lifecycle from experimentation to production deployment using industry best practices.

## 📋 Prerequisites

- Basic Python programming (pandas, scikit-learn)
- Understanding of machine learning concepts
- Familiarity with Git version control
- Basic command line knowledge
- Docker basics (containers, images, Dockerfile)

## 🎓 Learning Objectives

By the end of this course, students will be able to:
- Implement end-to-end MLOps pipelines using MLFlow
- Track experiments, manage models, and deploy ML applications
- Apply MLOps best practices for GenAI workflows
- Build production-ready ML systems with proper monitoring and testing
- Collaborate effectively in ML teams using MLFlow and modern tools
- Version control data and models using DVC
- Implement CI/CD pipelines for ML projects
- Apply security and governance best practices in MLOps

---

## 🔧 Additional Topics Beyond MLFlow Documentation

This course covers essential MLOps topics that complement the official MLFlow documentation:

### **Data Engineering & Quality**
- **Data Validation**: Quality checks, profiling, and drift detection
- **Dataset Management**: Versioning datasets within MLFlow artifacts
- **Feature Engineering**: Best practices and reproducibility

### **Software Engineering for ML**
- **Testing Strategies**: Unit tests, integration tests, and ML-specific testing
- **Code Quality**: Linting (ruff), type checking (mypy), and pre-commit hooks
- **Environment Management**: Modern Python tooling with uv

### **DevOps & Infrastructure**
- **CI/CD for ML**: Automated testing, validation, and deployment pipelines
- **Containerization**: Docker best practices for ML applications
- **Security**: Authentication, authorization, and secure deployments

### **Production Operations**
- **Monitoring**: Model performance, data drift, and system health
- **Cost Management**: Resource optimization and budget tracking
- **Incident Response**: Debugging and troubleshooting ML systems

### **Team Collaboration**
- **Governance**: Model approval workflows and compliance
- **Documentation**: Technical documentation and knowledge sharing
- **Project Management**: Agile methodologies for ML projects

### **Business Integration**
- **ROI Measurement**: Quantifying MLOps value and impact
- **Stakeholder Communication**: Presenting technical concepts to business users
- **Change Management**: Adoption strategies and organizational transformation

These topics ensure students gain comprehensive MLOps knowledge beyond just MLFlow usage, preparing them for real-world enterprise environments.

---

## 📅 Session 1: MLOps Fundamentals & Environment Setup (4 hours)

### Theory (1.5 hours)
- **MLOps Introduction** (30 min)
  - What is MLOps and why it matters
  - ML lifecycle challenges vs traditional software
  - MLOps maturity levels and adoption strategies
  - ROI and business impact of MLOps
  
- **MLFlow Ecosystem Overview** (45 min)
  - MLFlow components: Tracking, Projects, Models, Registry
  - Architecture and deployment options
  - Integration with cloud platforms and other tools
  - MLFlow vs alternatives comparison
  
- **Development Environment Best Practices** (15 min)
  - Modern Python environment management
  - Version control strategies for ML projects
  - Reproducibility principles and challenges

### Practical Lab (2.5 hours)
- **Environment Setup with uv** (45 min)
  ```bash
  # Install uv (fast Python package manager)
  curl -LsSf https://astral.sh/uv/install.sh | sh
  
  # Create MLOps course project
  uv init mlops-course
  cd mlops-course
  
  # Add all required dependencies
  uv add mlflow[extras] scikit-learn pandas numpy matplotlib seaborn
  uv add pytest python-dotenv fastapi uvicorn --group dev
  uv add pre-commit black ruff mypy --group dev
  
  # Activate environment and verify
  uv sync
  source .venv/bin/activate
  mlflow --version
  
  # Start MLFlow UI
  mlflow ui --host 0.0.0.0 --port 5000
  ```

- **Project Structure & Git Setup** (30 min)
  ```
  mlops-course/
  ├── .env                    # Environment variables
  ├── .gitignore             # Git ignore rules
  ├── pyproject.toml         # uv project configuration
  ├── README.md              # Project documentation
  ├── src/
  │   ├── __init__.py
  │   ├── data/              # Data processing modules
  │   ├── models/            # Model training modules
  │   ├── evaluation/        # Model evaluation
  │   └── utils/             # Utility functions
  ├── notebooks/             # Jupyter notebooks
  ├── tests/                 # Unit tests
  ├── docker/                # Docker configurations
  └── .github/
      └── workflows/         # GitHub Actions
  ```

- **First MLFlow Experiment** (75 min)
  - Create a comprehensive ML pipeline with scikit-learn
  - Log parameters, metrics, artifacts, and model metadata
  - Explore MLFlow UI in depth
  - Compare multiple runs and analyze results
  - Implement basic data validation and model testing

### 🛠️ Hands-on Project
Create a basic classification model with MLFlow tracking:
```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Start MLFlow experiment
mlflow.set_experiment("iris-classification")

with mlflow.start_run():
    # Load data
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Log parameters and metrics
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy_score(y_test, model.predict(X_test)))
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
```

---

## 📅 Session 2: Advanced MLFlow Features & Tool Selection Strategy (4 hours)

### Theory (1.5 hours)
- **MLFlow Model Registry Deep Dive** (30 min)
  - Model lifecycle management strategies
  - Staging environments (Development, Staging, Production)
  - Model promotion workflows and approval processes
  - Model lineage and metadata management

- **Data Management Strategy: MLFlow vs DVC** (45 min)
  - **When MLFlow artifacts are sufficient:**
    - Datasets < 500MB
    - Simple preprocessing pipelines
    - Small teams (< 5 data scientists)
    - Prototype and learning environments
  
  - **When DVC becomes necessary:**
    - Large datasets (> 1GB)
    - Complex multi-stage data pipelines
    - Multiple teams sharing datasets
    - Data-heavy organizations with frequent updates
  
  - **Decision Framework:**
    | Criteria | MLFlow Artifacts | DVC + MLFlow | Both |
    |----------|------------------|--------------|------|
    | **Data Size** | < 500MB | > 1GB | Mixed |
    | **Pipeline Complexity** | Simple transforms | Multi-stage pipelines | Varies |
    | **Team Size** | 1-5 people | 5+ people | Large orgs |
    | **Storage Cost** | Low concern | High concern | Critical |
    | **Learning Curve** | Minimal | Moderate | Complex |
    
  - **Real-world examples and case studies**
  - **Migration strategies**: Starting with MLFlow, when to add DVC

- **Testing Strategies for ML** (15 min)
  - Unit testing for ML code vs traditional software
  - Data validation and quality checks
  - Model testing strategies (performance, fairness, robustness)

### Practical Lab (2.5 hours)
- **Advanced MLFlow Model Registry** (75 min)
  ```python
  # Advanced model registration with metadata
  import mlflow
  from mlflow.tracking import MlflowClient
  
  client = MlflowClient()
  
  # Register model with rich metadata
  model_version = mlflow.register_model(
      model_uri=f"runs:/{run.info.run_id}/model",
      name="customer-churn-predictor",
      tags={
          "team": "data-science",
          "algorithm": "random_forest",
          "performance_threshold": "0.85",
          "data_version": "2024_Q1",
          "approved_by": "senior_ds"
      }
  )
  
  # Demonstrate data tracking with MLFlow
  with mlflow.start_run():
      # Method 1: Log datasets as artifacts
      mlflow.log_artifact("data/train.csv", "datasets")
      mlflow.log_artifact("data/test.csv", "datasets")
      
      # Method 2: Use MLFlow dataset tracking
      dataset = mlflow.data.from_pandas(df, source="internal_db", name="customer_data")
      mlflow.log_input(dataset, context="training")
      
      # Method 3: Track data lineage with tags
      mlflow.set_tag("data_source", "customer_db_v2")
      mlflow.set_tag("feature_engineering", "v3.1")
      mlflow.set_tag("data_quality_score", "95%")
  ```

- **Tool Selection Exercise** (30 min)
  - **Scenario Analysis**: Students receive 3 different company scenarios
    - Startup with small datasets
    - Mid-size company with growing data needs  
    - Enterprise with complex data pipelines
  - **Group Discussion**: Which approach (MLFlow only vs MLFlow + DVC) for each scenario
  - **Decision Documentation**: Students document their reasoning

- **Comprehensive ML Testing** (75 min)
  ```python
  # pytest examples for ML systems
  import pytest
  import pandas as pd
  import numpy as np
  from src.models.train import load_model, preprocess_data
  
  def test_data_quality():
      """Test data integrity and quality"""
      data = load_data()
      assert not data.isnull().any().any(), "Data contains null values"
      assert len(data) > 1000, "Insufficient training data"
      assert data['target'].nunique() > 1, "No class variation in target"
  
  def test_model_performance():
      """Test model meets performance requirements"""
      model = load_model()
      X_test, y_test = load_test_data()
      accuracy = model.score(X_test, y_test)
      assert accuracy > 0.85, f"Model accuracy {accuracy} below threshold"
  
  def test_data_lineage_tracking():
      """Test that data versions are properly tracked"""
      with mlflow.start_run() as run:
          # Verify data tracking
          data_tags = mlflow.get_run(run.info.run_id).data.tags
          assert "data_source" in data_tags, "Data source not tracked"
          assert "data_version" in data_tags, "Data version not tracked"
  ```

### 🛠️ Hands-on Project
Build a hyperparameter optimization pipeline:
```python
import mlflow
import optuna
from mlflow.tracking import MlflowClient

def objective(trial):
    with mlflow.start_run(nested=True):
        # Suggest parameters
        n_estimators = trial.suggest_int('n_estimators', 10, 100)
        max_depth = trial.suggest_int('max_depth', 1, 10)
        
        # Train and evaluate model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth
        )
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        
        # Log to MLFlow
        mlflow.log_params(trial.params)
        mlflow.log_metric("accuracy", accuracy)
        
        return accuracy

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
```

---

## 📅 Session 3: CI/CD, Security & Production Deployment (4 hours)

### Theory (1 hour)
- **CI/CD for MLOps** (30 min)
  - Differences between software CI/CD and ML CI/CD
  - Automated testing strategies for ML pipelines
  - Model validation gates and approval processes
  - Rollback strategies for ML models
  
- **Security & Governance** (30 min)
  - Model security considerations
  - Data privacy and compliance (GDPR, CCPA)
  - Model governance and audit trails
  - Access control and authentication in MLFlow

### Practical Lab (3 hours)
- **GitHub Actions for MLOps** (90 min)
  ```yaml
  # .github/workflows/ml-pipeline.yml
  name: ML Pipeline
  on:
    push:
      branches: [main]
    pull_request:
      branches: [main]

  jobs:
    test:
      runs-on: ubuntu-latest
      steps:
      - uses: actions/checkout@v4
      
      - name: Install uv
        uses: astral-sh/setup-uv@v1
        
      - name: Install dependencies
        run: uv sync
        
      - name: Run tests
        run: |
          uv run pytest tests/
          uv run ruff check .
          uv run mypy src/
          
      - name: Data validation
        run: uv run python src/validation/validate_data.py
        
    deploy:
      needs: test
      if: github.ref == 'refs/heads/main'
      runs-on: ubuntu-latest
      steps:
      - name: Train and register model
        run: uv run python src/models/train.py
        
      - name: Deploy to staging
        run: uv run python src/deployment/deploy.py --environment staging
  ```

- **Secure Docker Deployment** (90 min)
  ```dockerfile
  # Multi-stage secure Docker build
  FROM python:3.11-slim as builder
  RUN pip install uv
  COPY pyproject.toml uv.lock ./
  RUN uv sync --frozen --no-dev

  FROM python:3.11-slim as production
  # Security: non-root user
  RUN groupadd -r mluser && useradd -r -g mluser mluser
  
  # Copy only necessary files
  COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
  COPY src/ /app/src/
  
  USER mluser
  WORKDIR /app
  
  # Health check
  HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
    
  EXPOSE 8000
  CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
  ```

### 🛠️ Hands-on Project
Create a complete MLFlow project with deployment:

**MLproject file:**
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

**Deployment:**
```bash
# Build and serve model
mlflow models build-docker -m "models:/iris-model/1" -n "iris-model"
docker run -p 5000:8080 iris-model
```

---

## 📅 Session 4: GenAI MLOps & Cost Optimization (4 hours)

### Theory (1 hour)
- **GenAI MLOps Challenges** (30 min)
  - LLM lifecycle management complexities
  - Prompt engineering and versioning strategies
  - Evaluation metrics for GenAI (BLEU, ROUGE, custom metrics)
  - Fine-tuning vs RAG vs prompt engineering trade-offs
  
- **Cost Management & Optimization** (30 min)
  - LLM cost monitoring and budgeting
  - Token usage optimization techniques
  - Model selection based on cost-performance
  - Caching strategies for LLM responses

### Practical Lab (3 hours)
- **LLM Experiment Tracking & Cost Control** (90 min)
  ```python
  # Cost-aware LLM tracking
  import mlflow
  import time
  from functools import wraps

  def track_llm_costs(func):
      @wraps(func)
      def wrapper(*args, **kwargs):
          start_time = time.time()
          result = func(*args, **kwargs)
          
          # Calculate costs
          input_tokens = result.get('usage', {}).get('prompt_tokens', 0)
          output_tokens = result.get('usage', {}).get('completion_tokens', 0)
          cost = calculate_cost(input_tokens, output_tokens, model_name)
          
          # Log to MLFlow
          mlflow.log_metric("input_tokens", input_tokens)
          mlflow.log_metric("output_tokens", output_tokens)
          mlflow.log_metric("cost_usd", cost)
          mlflow.log_metric("latency_seconds", time.time() - start_time)
          
          return result
      return wrapper
  
  @track_llm_costs
  def call_llm(prompt, model="gpt-3.5-turbo"):
      # LLM call implementation
      pass
  ```

- **Advanced GenAI Pipeline** (90 min)
  - Prompt template versioning and A/B testing
  - RAG (Retrieval Augmented Generation) implementation
  - Custom evaluation metrics for domain-specific tasks
  - Multi-model comparison and selection
  - Response caching and optimization

### 🛠️ Hands-on Project
Build a production-ready GenAI pipeline:
```python
# Complete RAG system with MLFlow tracking
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
            # Log prompt components
            mlflow.log_text(query, "user_query.txt")
            mlflow.log_text(context, "retrieved_context.txt")
            mlflow.log_text(response, "generated_response.txt")
            
            # Log metrics
            mlflow.log_metric("context_relevance", self.calculate_relevance(query, context))
            mlflow.log_metric("response_quality", self.evaluate_response(response))
            
            # Log costs and performance
            mlflow.log_metric("retrieval_time", self.retrieval_time)
            mlflow.log_metric("generation_cost", self.generation_cost)
```

---

## 📅 Session 5: Production Operations & Enterprise MLOps (4 hours)

### Theory (1 hour)
- **Production Debugging & Troubleshooting** (30 min)
  - Common production issues and solutions
  - Debugging ML model performance degradation
  - Log analysis and error tracking
  - Incident response for ML systems
  
- **Team Collaboration & Governance** (30 min)
  - MLOps team structures and responsibilities
  - Model governance frameworks
  - Compliance and audit requirements
  - Change management for ML systems

### Practical Lab (3 hours)
- **Production Monitoring & Alerting** (90 min)
  ```python
  # Production monitoring with custom metrics
  import mlflow
  import prometheus_client
  from datadog import statsd
  
  class ProductionMonitor:
      def __init__(self):
          self.model_accuracy = prometheus_client.Gauge(
              'ml_model_accuracy', 'Current model accuracy'
          )
          self.prediction_latency = prometheus_client.Histogram(
              'ml_prediction_latency_seconds', 'Prediction latency'
          )
          
      def log_prediction(self, features, prediction, actual=None):
          start_time = time.time()
          
          # Make prediction
          result = self.model.predict(features)
          
          # Log to MLFlow
          with mlflow.start_run():
              mlflow.log_metric("prediction_latency", time.time() - start_time)
              if actual is not None:
                  accuracy = calculate_accuracy(result, actual)
                  mlflow.log_metric("live_accuracy", accuracy)
                  
              # Check for data drift
              drift_score = self.detect_drift(features)
              if drift_score > 0.1:
                  self.send_alert("Data drift detected", drift_score)
  ```

- **Enterprise Integration & Governance** (90 min)
  - Role-based access control (RBAC) setup
  - Model approval workflows
  - Compliance reporting and audit trails
  - Integration with enterprise tools (LDAP, SSO)
  - Multi-environment promotion (dev → staging → prod)
  
  ```bash
  # Enterprise deployment with uv and Docker
  # Production environment setup
  uv run mlflow server \
    --backend-store-uri postgresql://user:pass@db:5432/mlflow \
    --default-artifact-root s3://mlflow-artifacts \
    --host 0.0.0.0 \
    --port 5000 \
    --serve-artifacts
  
  # Model promotion workflow
  uv run python scripts/promote_model.py \
    --model-name "customer-churn" \
    --version 5 \
    --stage "Production" \
    --approval-required
  ```

### 🛠️ Final Capstone Project
Complete enterprise-grade MLOps system:

**Project Requirements:**
- End-to-end ML pipeline with uv environment management
- Data versioning with DVC
- Comprehensive testing suite (unit, integration, performance)
- CI/CD with GitHub Actions
- Secure Docker deployment
- Production monitoring and alerting
- Cost tracking and optimization
- Team collaboration features
- Governance and compliance documentation

**Deliverables:**
1. **Technical Implementation** (Git repository)
   - Complete source code with uv configuration
   - Docker multi-stage builds
   - GitHub Actions workflows
   - Comprehensive test suite
   - Documentation and README

2. **Production Deployment** (Live demo)
   - Running MLFlow server with model registry
   - Deployed model API with monitoring
   - Dashboard showing metrics and costs
   - Alerting system demonstration

3. **Team Presentation** (15 min + 5 min Q&A)
   - Architecture overview and design decisions
   - Live demonstration of the pipeline
   - Lessons learned and challenges overcome
   - Production readiness assessment

---

## 📊 Assessment & Evaluation

### Individual Assignments (50%)
- Session experiments and labs (30%)
- Code quality and documentation (10%)
- MLFlow best practices implementation (10%)

### Group Project & Presentation (50%)
- **End-to-end MLOps pipeline development (30%)**
  - Team collaboration using MLFlow
  - Production deployment implementation
  - Code quality and documentation
  
- **Team Presentation & Demo (20%)**
  - 15-minute presentation per team
  - Live demonstration of the MLOps pipeline
  - Q&A session with instructors and peers
  - Technical depth and clarity of explanation

### Presentation Guidelines
- **Duration**: 15 minutes presentation + 5 minutes Q&A
- **Format**: Live demo with slides support
- **Content Requirements**:
  - Problem statement and solution approach
  - MLFlow implementation walkthrough
  - Live pipeline demonstration
  - Challenges faced and solutions implemented
  - Lessons learned and best practices
- **Evaluation Criteria**:
  - Technical accuracy and implementation quality
  - Clarity of presentation and communication
  - Team collaboration demonstration
  - Innovation and best practices application

---

## 🛠️ Tech Stack & Tools

### Core MLOps Tools (Open Source)
- **MLFlow** - Experiment tracking, model registry, deployment
- **Python 3.8+** - Primary programming language
- **GitHub** - Version control and collaboration
- **GitHub Actions** - CI/CD automation
- **Docker** - Containerization and deployment

### Supporting Tools
- **DVC** - Data version control and pipeline management
- **pytest** - Unit testing for ML code
- **FastAPI** - Custom model serving APIs
- **python-dotenv** - Environment configuration
- **pre-commit** - Code quality automation
- **SQLite/PostgreSQL** - MLFlow backend storage

### Development Environment
- **uv** - Fast Python package and environment manager
- **VS Code** (recommended IDE)
- **Git** - Version control client
- **Docker Desktop** - Container management

### Environment Setup
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create project environment
uv init mlops-course
cd mlops-course
uv add mlflow scikit-learn pandas docker pytest dvc fastapi

# Activate environment
uv sync
```

## 📚 Required Resources

### Official Documentation
- [MLFlow ML Guide](https://mlflow.org/docs/latest/ml/)
- [MLFlow GenAI Guide](https://mlflow.org/docs/latest/genai/)
- [DVC Documentation](https://dvc.org/doc)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

### Environment Setup
```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create course project
uv init mlops-course
cd mlops-course

# Add core dependencies
uv add mlflow[extras] scikit-learn pandas numpy matplotlib seaborn
uv add dvc fastapi uvicorn python-dotenv

# Add development tools
uv add pytest black ruff mypy pre-commit --group dev

# Add optional ML libraries
uv add optuna chromadb sentence-transformers --group ml

# Activate environment
uv sync
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows
```

### Sample Datasets & APIs
- **Traditional ML**: Iris, Boston Housing, Titanic, Wine Quality
- **Time Series**: Stock prices, IoT sensor data, sales forecasting
- **NLP/GenAI**: Movie reviews, customer support tickets, documentation
- **Computer Vision**: CIFAR-10, Fashion-MNIST (if time permits)

### Cloud Resources (Optional)
- **Free tiers**: AWS, Google Cloud, Azure for deployment demos
- **Alternatives**: Local Docker containers for full offline experience

---

## 🎯 Post-Course Learning Path

### Intermediate Topics
- MLFlow at scale with Kubernetes
- Custom MLFlow plugins development
- Integration with Apache Airflow
- Multi-cloud MLOps strategies

### Advanced Specializations
- MLOps for Computer Vision
- Real-time ML with streaming data
- Federated learning workflows
- MLOps governance and compliance

---

## 🤝 Support & Community

- Course forum for Q&A
- Office hours with instructors
- MLFlow community resources
- Industry guest speakers

**Certificate:** Upon successful completion, participants receive a "MLOps with MLFlow Professional" certificate.
