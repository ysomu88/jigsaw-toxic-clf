import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train import load_model
from src.preprocess import clean_text

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
app = FastAPI(title="Toxic Comment Classifier", version="1.0.0")

try:
    model, vectorizer = load_model()
except Exception:
    model = None
    vectorizer = None

class CommentRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    text: str
    predictions: dict
    is_toxic: bool

@app.get("/")
def root():
    return {"status": "ok", "message": "Toxic Comment Classifier API", "model_loaded": model is not None}

@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}

@app.post("/predict")
def predict(request: CommentRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run the training pipeline first.")
    
    cleaned_text = clean_text(request.text)
    features = vectorizer.transform([cleaned_text])
    preds = model.predict(features)
    predictions = {label: int(preds[0][i]) for i, label in enumerate(LABELS)}
    is_toxic = any(v == 1 for v in predictions.values())

    return PredictionResponse(text=request.text, predictions=predictions, is_toxic=is_toxic)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

