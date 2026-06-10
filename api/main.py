import os
import sys
from typing import List, Dict
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

def clean_text(text: str) -> str:  # noqa: E501
    """Clean the input text by converting to lowercase and removing extra whitespace."""
    return ' '.join(str(text).lower().split())


LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]  # noqa: F811

from src.train import load_model
model, vectorizer = load_model()


app = FastAPI(title="Toxic Comment Classifier")


class CommentRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    text: str
    predictions: Dict[str, int]  # noqa: F812
    is_toxic: bool

@app.get("/")
async def root():
    return {"status": "ok", "message": "Toxic Comment Classifier API"}


def vectorize_and_predict(text):
    cleaned_text = clean_text(text)
    
    features = vectorizer.transform([cleaned_text])  # noqa: E501
    
    preds_list = model.predict(features)[0]
    return {LABELS[i]: int(preds_val) for i, preds_val in enumerate(preds_list)}


@app.post("/predict")
async def predict(request: CommentRequest):
    if not request.text.strip():  # noqa: E712 (explicit check for empty string is preferred over falsy checks with .strip())
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")
    
    predictions = vectorize_and_predict(request.text)

    return PredictionResponse(
        text=request.text, 
        predictions=predictions, 
        is_toxic=True if any(pred_val == 1 for pred_val in predictions.values()) else False # noqa: E712
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


