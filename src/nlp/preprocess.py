import re
import pandas as pd
from unidecode import unidecode
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#\w+")
NON_ALPHANUMERIC_PATTERN = re.compile(r"[^a-zA-Z0-9\s]")


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = unidecode(text)
    text = text.lower()
    text = URL_PATTERN.sub("", text)     # Remoção de URls
    text = MENTION_PATTERN.sub("", text)     # Remoção de Menções
    text = HASHTAG_PATTERN.sub("", text)     # Remoção de Hashtags
    text = NON_ALPHANUMERIC_PATTERN.sub("", text)     # Remoção de Pontuação
    tokens = nltk.word_tokenize(text)     # Tokenização
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]     # Remoção de Stopwords
    tokens = [lemmatizer.lemmatize(t) for t in tokens]     # Lematização
    return " ".join(tokens)

def preprocess_dataframe(df: pd.DataFrame, text_col = 'text', label_col = None):
    df = df.copy()
    df['text_clean'] = df[text_col].apply(clean_text)
    if label_col is not None:
        df = df[[ 'text', 'text_clean', label_col ]].dropna(subset=['text_clean'])
    else:
        df = df[['text','text_clean']].dropna(subset=['text_clean'])
    return df 