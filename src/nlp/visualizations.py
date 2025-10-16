from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io


# função para gerar wordcloud como bytes
def gerar_wordcloud_bytes(sentimento: str = 'geral') -> bytes:
    # placeholder: em produção, filtre textos por sentimento
    sample_text = 'python dados machine learning visualizacao wordcloud sentimento tweets analise'
    wc = WordCloud(width=800, height=400, background_color='white').generate(sample_text)
    buf = io.BytesIO()
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf.read()