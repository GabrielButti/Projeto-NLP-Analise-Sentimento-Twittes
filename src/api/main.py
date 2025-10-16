from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from src.api.routes import router


app = FastAPI(title="API de Análise de Sentimento")
app.include_router(router, prefix="")


@app.get("/", response_class=HTMLResponse)
def root():
    return "<h3>API de Análise de Sentimento - veja /docs para testar</h3>"


@app.get("/health")
def health():
    return {"status":"ok"}