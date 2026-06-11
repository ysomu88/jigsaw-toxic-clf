# Jigsaw Toxic Comment Classification

A production-grade machine learning pipeline for detecting toxic language in text comments. This repository implements a multi-label classification system trained on the Jigsaw Toxic Comment dataset, with experiment tracking via MLflow and a FastAPI serving layer for real-time inference.

## Tech Stack

- **Python 3.10+**
- **FastAPI** - High-performance async web framework
- **MLflow** - Experiment tracking and model registry
- **uv** - Lightning-fast Python package manager and virtual environment tool
- **scikit-learn** - Machine learning models (LogisticRegression with OneVsRestClassifier)

---

## Repository Architecture

```
jigsaw-toxic-clf/
├── api/
│   ├── __init__.py
│   └── main.py              # FastAPI serving layer
├── models/
│   ├── model.pkl            # Trained classifier
│   ├── vectorizer.pkl       # TF-IDF vectorizer
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Dataset loading utilities
│   ├── preprocess.py        # Text cleaning and TF-IDF vectorization
│   ├── train.py             # Model training with MLflow tracking
│   └── evaluate.py          # Model evaluation metrics
├── run_pipeline.py          # Main pipeline orchestrator
├── requirements.txt         # Python dependencies
├── mlflow.db                # SQLite MLflow backend (if using file store)
└── mlruns/                  # MLflow runs directory
```

---

## Local Environment Setup

### Prerequisites

- **Windows PowerShell** (5.1 or later)
- **uv** installed globally (https://github.com/astral-sh/uv)

### Step 1: Configure Execution Policy

If you encounter permission errors, run PowerShell as Administrator and execute:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 2: Create Virtual Environment

```powershell
uv venv .venv
```

### Step 3: Activate Environment

```powershell
# PowerShell
.venv\Scripts\Activate.ps1

# Or for CMD
.venv\Scripts\activate.bat
```

### Step 4: Install Dependencies

```powershell
uv pip install -r requirements.txt
```

**Optional:** Install additional tools for Kaggle dataset access:

```powershell
uv pip install kaggle
```

---

## Model Training & Experiment Tracking

### Running the Training Pipeline

Execute the complete ML pipeline from the project root:

```powershell
python run_pipeline.py
```

**Pipeline Steps:**
1. **Data Loading** - Loads the Jigsaw Toxic Comment dataset
2. **Data Splitting** - Splits into train/test sets (80/20)
3. **Preprocessing** - Applies TF-IDF vectorization with ngram_range=(1,2)
4. **Model Training** - Trains LogisticRegression with OneVsRestClassifier
5. **Model Saving** - Persists model and vectorizer to `models/` directory
6. **Evaluation** - Computes F1 scores for all toxicity labels

### MLflow Experiment Tracking

The pipeline automatically logs:
- **Parameters**: model_type, tfidf_max_features, tfidf_ngram_range, lr_C, lr_max_iter, test_size, lr_solver
- **Metrics**: F1 scores for each label (toxic, severe_toxic, obscene, threat, insult, identity_hate) + macro_f1
- **Artifacts**: Trained model and vectorizer

### Viewing MLflow Dashboard

Start the local MLflow server to review experiments:

```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db
```

Then open your browser to: `http://localhost:5000`

**Alternative:** If using the file store backend, MLflow runs are stored in the `mlruns/` directory.

---

## API Serving Layer

### Starting the FastAPI Server

```powershell
python api/main.py
```

**Default Configuration:**
- **Host**: `0.0.0.0`
- **Port**: `8000`
- **Access URL**: `http://localhost:8000`

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check and API info |
| `/health` | GET | Model health status |
| `/predict` | POST | Text toxicity inference |

### Testing the API

#### Using PowerShell (Invoke-RestMethod)

```powershell
# Test the toxicity prediction endpoint
$body = @{
    text = "This is a normal comment without any toxic language."
} | ConvertTo-Json

$response = Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method POST -Body $body -ContentType "application/json"

# Display results
$response | ConvertTo-Json
```

#### Using curl (Windows PowerShell)

```powershell
$body = @{
    text = "You are an idiot and I hate you!"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method POST -Body $body -ContentType "application/json"
```

#### Expected Response Format

```json
{
  "text": "You are an idiot and I hate you!",
  "predictions": {
    "toxic": 1,
    "severe_toxic": 1,
    "obscene": 1,
    "threat": 0,
    "insult": 1,
    "identity_hate": 0
  },
  "is_toxic": true
}
```

### API Response Fields

- **text**: The original input text
- **predictions**: Dictionary with binary predictions (0/1) for each toxicity label
- **is_toxic**: Boolean indicating if ANY label is positive (any toxicity detected)

---

## Model Labels

The classifier detects six types of toxicity:

| Label | Description |
|-------|-------------|
| `toxic` | General toxicity |
| `severe_toxic` | Severe toxicity |
| `obscene` | Obscene language |
| `threat` | Threatening content |
| `insult` | Insulting language |
| `identity_hate` | Identity-based hate speech |

---

## Troubleshooting

### Model Not Loaded

If the API returns "Model not loaded" errors:

```powershell
# Re-run the training pipeline
python run_pipeline.py

# Verify model files exist
ls models\
```

### MLflow Server Not Starting

Ensure no other MLflow server is running on port 5000:

```powershell
# Check for existing processes
Get-Process | Where-Object {$_.ProcessName -like "*python*"}

# Kill existing processes if needed
Get-Process python | Stop-Process -Force
```

### Permission Errors

Run PowerShell as Administrator and set execution policy:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## License

This project is provided for educational purposes.
