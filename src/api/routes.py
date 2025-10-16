from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import io
import pandas as pd
from src.api.services import analyze_text, analyze_batch_csv_bytes
from src.api.utils import read_twitter_csv
from pydantic import BaseModel


router = APIRouter()


class TextoEntrada(BaseModel):
    texto: str

@router.post('/analise/')
def rota_analise(payload: TextoEntrada):
    if not payload.texto:
        raise HTTPException(status_code=400, detail="Campo 'texto' vazio")

    try:
        result = analyze_text(payload.texto)  # já é dict com 'prediction' e 'probabilities'
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao analisar texto: {e}")

@router.post('/analise/lote/')
async def rota_analise_lote(file: UploadFile = File(...)):
    try:
        content = await file.read()  # lê bytes
        df = read_twitter_csv(content)  # usa a função utilitária que já define as colunas
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Falha ao ler CSV: {e}")

    # o read_twitter_csv garante que a coluna 'text' exista
    if 'text' not in df.columns:
        raise HTTPException(status_code=400, detail="CSV deve conter a coluna 'text'")

    results = analyze_batch_csv_bytes(content)  # envia os bytes, não o df
    return JSONResponse(content=results)
