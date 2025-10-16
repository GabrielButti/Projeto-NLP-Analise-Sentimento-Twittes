from pydantic import BaseModel


class AnaliseRequest(BaseModel):
    texto: str


class AnaliseResponse(BaseModel):
    texto: str
    sentimento: str
    probabilidade: list