import pandas as pd
import io

def read_twitter_csv(file_bytes: bytes) -> pd.DataFrame:
    """
    Lê um CSV simples contendo apenas frases (uma por linha),
    todas envoltas em aspas duplas.
    Retorna DataFrame com coluna única 'text'.
    """
    s = file_bytes.decode('utf-8', errors='replace')
    df = pd.read_csv(
        io.StringIO(s),
        header=None,
        names=['text'],
        quotechar='"',   # trata as aspas duplas
        skip_blank_lines=True
    )
    df['text'] = df['text'].astype(str)
    return df[['text']]
