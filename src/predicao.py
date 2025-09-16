import pandas as pd
import joblib
from pre_processamento import preprocess_dataframe

def predict_file(csv_path, out_path="Projeto-NPL-Analise-Sentimento-Twittes/data/generate/predictions.csv"):
    model = joblib.load("Projeto-NPL-Analise-Sentimento-Twittes/models/modelo_sentimento.pkl")
    columns = ['id','subject','sentiment','text']
    df = pd.read_csv(csv_path, names = columns)
    df = preprocess_dataframe(df, text_col='text', label_col='sentiment')

    # assume coluna 'text'
    preds = model.predict(df['text'])
    probs = model.predict_proba(df['text']) if hasattr(model, "predict_proba") else None
    df['sentiment_pred'] = preds
    if probs is not None:
        if probs.shape[1] == 2:  # binário
            df['prob_neg'] = probs[:,0]
            df['prob_pos'] = probs[:,1]
    df.to_csv(out_path, index=False)
    print("Predições salvas em", out_path)

# Exemplo de uso
if __name__ == "__main__":
    predict_file("Projeto-NPL-Analise-Sentimento-Twittes/data/raw/twitter_validation.csv")
