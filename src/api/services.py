from typing import List, Dict, Any
from src.nlp.sentiment_model import SentimentModel
from src.api.utils import read_twitter_csv
import pandas as pd

_model = SentimentModel()

def analyze_text(text: str) -> Dict[str, Any]:
    """
    Analisa uma frase e retorna o sentimento e as probabilidades.
    """
    # Garantir que o modelo recebe uma lista
    pred, probs = _model.predict(text)

    return {
        "prediction": pred,       # rótulo do sentimento
        "probabilities": probs    # lista de probabilidades
    }

def analyze_batch_csv_bytes(file_bytes: bytes) -> List[Dict[str, Any]]:
    df = read_twitter_csv(file_bytes)
    results = []
    for _, row in df.iterrows():
        text = row['text']
        pred, probs = _model.predict(text)
        res = {
            "text": text,
            "prediction": pred,
            "probabilities": probs
        }
        results.append(res)
    return results
