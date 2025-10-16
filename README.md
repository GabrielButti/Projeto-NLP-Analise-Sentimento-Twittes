# 📊 Análise de Sentimento (NLP) sobre Tweets

## 📌 Descrição
Projeto de **classificação de sentimento** aplicado a tweets. O objetivo é identificar **sentimentos** (positivo / neutro / negativo) em textos curtos, gerar **visualizações** (wordclouds) e disponibilizar um **pipeline** reprodutível para inferência em produção.


## 🎯 Objetivos da Análise
- **Construir um modelo de classificação** de sentimento com boa performance.
- **Gerar insights** sobre palavras mais frequentes por classe.
- **Disponibilizar um endpoint** simples para inferência.
- **Documentar o processo** para reprodutibilidade.

## ❓ Perguntas de Negócio
- Qual a distribuição de sentimento (positivo / neutro / negativo) na base?
- Quais palavras/termos aparecem mais em tweets positivos vs. negativos? (wordclouds)
- Qual a acurácia prática do modelo em identificar sentimento (Precision, Recall, F1 por classe)?
- O modelo é robusto a ruído: URLs, menções, emojis?
- Como a análise pode suportar ações (monitoramento de marca, detecção de crises, automação de respostas)?


## 🗂️ Estrutura do Projeto

```
projeto-nlp-sentimento/
├── data/
│   ├── raw/twitter_training.csv                # Arquivo CSV utilizado
│   ├── raw/twitter_validation.csv              # Arquivo CSV para validação
│   └── generate/predictions.csv                # Arquivo CSV com predições
├── notebooks/
│   ├── eda_nlp.ipynb                           # Notebook com análise exploratória
│   └── modelagem_nlp.ipynb                     # Notebook com modelagem
├── src/
│   ├── api/
│   │    ├── main.py                            # Inicia o servidor FastAPI
│   │    ├── routes.py                          # Define as rotas de inferência
│   │    ├── models.py                          # Carrega o modelo e faz predições
│   │    ├── services.py                        # Lógica de predição com o modelo
│   │    └── utils.py                           # Funções utilitárias para pré-processamento e form
│   ├── nlp/
│   │    ├── preprocess.py                      # Funções de pré-processamento de texto
│   │    ├── prediction.py                      # Função para fazer predições
│   │    ├── sentiment_model.py                 # Função para treinar e salvar o modelo
│   │    └── visualization.py                   # Funções para gerar visualizações (wordclouds)
│   └── ui/
│        └── app.py                             # Interface Streamlit  
├── models/
│   └── modelo_sentimento.pkl                   # Modelo treinado salvo
├── assets/
│   ├── curva_ROC.png                           # Curva ROC do modelo
│   ├── distribuicao_classes_sentimentos.png    # Distribuição das classes
│   ├── distribuicao_tamanho_tweet.png          # Distribuição do tamanho dos tweets
│   ├── matriz_confusao.png                     # Matriz de confusão do modelo
│   ├── nuvem_palavras_negativas.png            # Nuvem de palavras negativas
│   ├── nuvem_palavras_positivas.png            # Nuvem de palavras positivas
│   └── palavras_frequentes.png                 # Palavras mais frequentes
├── docker-compose.yml                          # Orquestra os containers API + UI
├── Dockerfile.api                              # Dockerfile da API FastAPI
├── Dockerfile.ui                               # Dockerfile da UI Streamlit
├── requirements.txt                            # Dependências do projeto 
└── README.md                                   # Documentação do projeto
```

## 🔧 Ferramentas Utilizadas
- **Python 3.14+**
- **Pandas / Numpy / Unidecode** – Manipulação de dados
- **Scikit-learn** – Modelagem preditiva
- **Joblib** – Salvamento do modelo
- **FastAPI / Unicorn** – Criação de API para inferência
- **WordCloud / NLTK** – Processamento de linguagem natural
- **Matplotlib / Seaborn / WordCloud** – Visualização de dados
- **Jupyter Notebook** – Documentação da análise
- **Streamlit** – Interface web para análise interativa
- **Docker / Docker Compose** – Containerização e orquestração
---

## 📊 Principais Insights

### Distribuição das Classes de Sentimento
- A base de dados apresenta uma distribuição relativamente equilibrada entre as classes de sentimento com exceção da classe **Irrelevante** que é a menos representada.:
  - **Positivo**: 27.8%
  - **Neutro**: 24.5%
  - **Negativo**: 30.1%
  - **Irrelevante**: 17.3%

- A classe **Negativo** é a mais frequente, seguida por **Positivo** e **Neutro**. A classe **Irrelevante** é a menos representada.


![Distribuição das Classes de Sentimento](assets/distribuicao_classes_sentimentos.png)

---

### Palavras Mais Frequentes em Sentimentos Positivos e Negativos
- **Positivo**: As palavras mais frequentes incluem "love", "great", "happy", "good", "amazing", "best", "fun", "awesome", "like", "thank".
- **Negativo**: As palavras mais frequentes incluem "hate", "bad", "sad", "angry", "terrible", "worst", "awful", "disappointed", "sucks", "annoyed".

![Nuvem de Palavras Positivas](assets/nuvem_palavras_positivas.png) 
![Nuvem de Palavras Negativas](assets/nuvem_palavras_negativas.png)

---

### Palavras Mais Frequentes no Dataset
- As palavras mais frequentes no dataset geral incluem "game", "like", "im", "get", "one", "play", "good", "time", "love", "really", "new".
![Palavras Mais Frequentes](assets/palavras_frequentes.png)

### Precisão do Modelo

#### **Acurácia**: 83%
#### **Recall**: 82%
#### **F1-score**: 82%

- O modelo apresenta uma **acurácia geral de 83%**, com bom equilíbrio entre precisão e recall para todas as classes.
- A classe **Negativo** tem a maior precisão (89%) e recall (84%), indicando que o modelo é eficaz em identificar tweets negativos.
- A classe **Irrelevante** tem a menor precisão (80%) e recall (80%), sugerindo que o modelo tem mais dificuldade em classificar corretamente tweets irrelevantes.

![Matriz de Confusão](assets/matriz_confusao.png)

---

### Modelo Robusto a Ruído
- O modelo mostrou-se robusto a ruídos comuns em tweets, como URLs, menções e emojis, graças ao pré-processamento eficaz.
- A remoção de URLs e menções, bem como a normalização de texto, ajudaram a melhorar a qualidade dos dados de entrada.
- Emojis foram convertidos em texto descritivo, permitindo que o modelo capturasse o sentimento associado a eles.
- A análise de erros indicou que a maioria dos erros de classificação ocorreu em tweets curtos ou ambíguos, onde o contexto é limitado.


![Distribuição do Tamanho dos Tweets](assets/distribuicao_tamanho_tweet.png)

---

### Suporte a Ações
- A análise de sentimento pode ser utilizada para **monitoramento de marca**, identificando rapidamente tweets negativos que possam indicar crises.
- Pode também ser usada para **automação de respostas**, direcionando tweets positivos para campanhas de engajamento e tweets negativos para atendimento ao cliente.
- A criação de dashboards interativos pode facilitar o acompanhamento em tempo real do sentimento dos tweets relacionados à marca ou produto.

![Curva ROC do Modelo](assets/curva_ROC.png)


---

## Próximos Passos
- **Avaliar modelos** baseados em transformers (BERT) para comparar performance.
- **Adicionar validação** temporal (se aplicável) e engenharia de features (emoji features, emoticons, presença de link).
- **Construir dashboard** em Streamlit que mostre volume de sentimento ao longo do tempo.
- **Implementar monitoramento** de modelo (drift) em produção.

## 📌 Como Reproduzir
```bash
git clone https://github.com/GabrielButti/Projeto-NLP-Analise-Sentimento-Twittes.git
cd Projeto-NLP-Analise-Sentimento-Twittes
pip install -r requirements.txt
python src/pre_processamento.py
jupyter notebook notebooks/modelagem_nlp.ipynb
python src/predicao.py
python src/api.py
uvicorn src.api:app --reload --port 8000
```

## 🚀 Novas Funcionalidades Implementadas

### 🧠 **API de Inferência (FastAPI)**
- Endpoints:
  - `POST /analise/` → analisa texto individual e retorna:
    - `prediction`: sentimento detectado.
    - `probabilities`: lista com 4 probabilidades (`Negative`, `Neutral`, `Mixed`, `Positive`).
  - `POST /analise/lote/` → aceita upload de CSV (com coluna `text`) e retorna análise em lote.
- Retorno em formato JSON, pronto para consumo via front-end ou integrações externas.

### 💻 **Interface Web (Streamlit UI)**
- Interface moderna e responsiva para uso direto no navegador.
- **Análise individual**: insira texto e veja o sentimento e probabilidades em tempo real.
- **Análise em lote (CSV)**: upload de arquivo e visualização dos resultados diretamente na tela.
- Download dos resultados com apenas um clique.
- Visualização das probabilidades em **gráficos de barras**.

### ☁️ **Nuvem de Palavras com Cores por Sentimento**
- Para a análise em lote, o sistema gera **wordclouds coloridas** por sentimento:
  - 🟢 **Positivo**
  - 🔴 **Negativo**
  - 🟠 **Neutro**
  - 🔵 **Misto**
- Permite análise visual rápida dos termos mais comuns associados a cada emoção.

### 🐳 **Containerização com Docker**
- Ambiente totalmente isolado e pronto para deploy local ou em nuvem.
- Containers separados:
  - `api` → serviço FastAPI
  - `ui` → interface Streamlit
- Orquestração via **docker-compose**:
  - Comunicação entre serviços via rede interna (`nlpnet`).
  - `API_URL` configurada automaticamente.
- Build otimizado com **cache inteligente** e instalação de dependências em camadas.

## 🧩 Como Executar com Docker

### 1️⃣ Build e subida dos containers

```bash
docker compose up --build
```
- API → disponível em http://localhost:8000

- UI → disponível em http://localhost:8501