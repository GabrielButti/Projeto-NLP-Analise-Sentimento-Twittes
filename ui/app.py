import streamlit as st
import pandas as pd
import requests
from io import BytesIO
import os

# Se rodando com docker-compose, API_URL pode ser 'http://api:8000'
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Dashboard Análise de Sentimentos", layout="wide")
st.title("Analise de Sentimentos (Apenas em Inglês)")

# --- SINGLE TEXT ANALYSIS ---
st.header("Análise Individual")
text = st.text_area("Digite alguma frase (Apenas em Inglês):")

if st.button("Analise"):
    if not text.strip():
        st.warning("Por favor insira algum texto.")
    else:
        try:
            resp = requests.post(f"{API_URL}/analise/", json={"texto": text})
            if resp.ok:
                out = resp.json()
                sentiment = out.get("prediction", "N/A")
                probs = out.get("probabilities", [])

                # --- Sentiment Highlight ---
                color_map = {
                    "Positive": "green",
                    "Negative": "red",
                    "Neutral": "orange",
                    "Mixed": "blue",
                }
                st.markdown(
                    f"<h2 style='text-align: center; color: {color_map.get(sentiment, 'black')};'>{sentiment}</h2>",
                    unsafe_allow_html=True
                )

                # --- Probabilities as bar chart ---
                if probs:
                    sentiments = ["Negative", "Neutral", "Mixed", "Positive"]
                    df_probs = pd.DataFrame({
                        "Sentiment": sentiments,
                        "Probability": probs
                    })
                    df_probs["Percentage"] = df_probs["Probability"] * 100
                    st.subheader("Probabilidades")
                    st.bar_chart(df_probs.set_index("Sentiment")["Percentage"])
                    st.table(df_probs.style.format({"Probability": "{:.2%}", "Percentage": "{:.2f}%"}))
                else:
                    st.info("No probability data available.")
            else:
                st.error(f"API error: {resp.text}")
        except Exception as e:
            st.error(f"Connection error: {e}")

# --- BATCH CSV ANALYSIS ---
st.markdown("---")
st.header("Analise de Sentimento em Lote via CSV")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    try:
        files = {"file": (uploaded.name, uploaded.getvalue(), "text/csv")}
        resp = requests.post(f"{API_URL}/analise/lote/", files=files)
        
        if resp.ok:
            results = resp.json()
            df = pd.DataFrame(results)
            st.subheader("Batch Results")
            st.dataframe(df)
            st.download_button("Download results", df.to_csv(index=False), "results.csv", "text/csv")
        else:
            st.error(f"API error: {resp.text}")
    except Exception as e:
        st.error(f"Connection error: {e}")


st.markdown("---")  # separador visual
st.markdown(
    "<small style='color:gray;'>⚠️ Disclaimer: Este modelo é automatizado e pode cometer erros. "
    "Os resultados fornecidos devem ser usados apenas como referência e não substituem análise humana.</small>",
    unsafe_allow_html=True
)