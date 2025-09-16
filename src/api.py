from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load(r"models/modelo_sentimento.pkl")

class TextIn(BaseModel):
    text: str

@app.post("/predict")
def predict(payload: TextIn):
    pred = model.predict([payload.text])[0]
    proba = model.predict_proba([payload.text])[0].tolist() if hasattr(model, "predict_proba") else None
    return {"text": payload.text, "prediction": pred, "probability": proba}
