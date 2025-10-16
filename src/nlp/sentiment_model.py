import os
import joblib
from typing import Tuple
from src.nlp.preprocess import clean_text
import os

# Caminho para o modelo
MODEL_PATH = os.getenv("MODEL_PATH")

if MODEL_PATH is None:
    # Detecta a raiz do projeto (uma pasta acima de src/)
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    MODEL_PATH = os.path.join(ROOT_DIR, "models", "modelo_sentimento.pkl")

class SentimentModel:
    def __init__(self):
        self.model = None
        self.classes_ = ["Negative", "Neutral", "Mixed", "Positive"]
        if os.path.exists(MODEL_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                print("Modelo carregado com sucesso!")
            except Exception as e:
                print(f"Falha ao carregar modelo: {e}")
                self.model = None
        else:
            print("Modelo não encontrado, usando fallback.")

    def predict(self, texto: str) -> Tuple[str, list]:
        texto_proc = clean_text(texto)

        if self.model is None or not hasattr(self.model, "predict_proba"):
            # fallback: distribui probabilidade total para o rótulo neutro
            pred = "Neutral"
            probs = [0.0, 1.0, 0.0, 0.0]
            return pred, probs

        # Faz predição
        pred = self.model.predict([texto_proc])[0]

        # Probabilidades do modelo
        prob_raw = self.model.predict_proba([texto_proc])[0]
        model_classes = self.model.classes_  # classes do modelo

        # Criar lista de probabilidades na ordem fixa
        probs = []
        for c in self.classes_:
            if c in model_classes:
                idx = list(model_classes).index(c)
                probs.append(prob_raw[idx])
            else:
                probs.append(0.0)  # se a classe não existir no modelo

        return pred, probs