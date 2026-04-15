from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

MODEL_PATH = "../distilbert_toxicity_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()


class Request(BaseModel):
    text: str


@app.get("/")
def root():
    return {"message": "Toxicity Detection API is running"}


@app.post("/predict")
def predict(request: Request):
    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()

    label = "Toxic" if prediction == 1 else "Non-Toxic"

    return {
        "input_text": request.text,
        "prediction": label,
        "class_id": prediction,
        "confidence": round(confidence, 4)
    }