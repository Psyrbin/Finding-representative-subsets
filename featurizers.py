from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import tensorflow_hub as hub

def cvec(data):
    stp_wrds = ['a', 'an', 'the', 'of', 'and', 'but', 'or', 'of', 'to']
    vectorizer = CountVectorizer(stop_words=stp_wrds, ngram_range=(1, 4))

    return vectorizer.fit_transform(data)

def tfidf(data):
    stp_wrds = ['a', 'an', 'the', 'of', 'and', 'but', 'or', 'of', 'to']
    vectorizer = TfidfVectorizer(stop_words=stp_wrds, ngram_range=(1, 3))

    return vectorizer.fit_transform(data)

def roberta(data):
    sbert_model = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')

def bert(data):
    sbert_model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
    return sbert_model.encode(data.tolist())

def distilroberta(data):
    sbert_model = SentenceTransformer('distilroberta-base-paraphrase-v1')
    return sbert_model.encode(data.tolist())

def glove(data):
    sbert_model = SentenceTransformer('average_word_embeddings_glove.6B.300d')
    return sbert_model.encode(data.tolist())

def universal(data):
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    model = hub.load(module_url)
    sentence_list = data.tolist()
    sentence_embeddings = []
    for i in range(len(sentence_list)):
        sentence_embeddings.append(np.array(model([sentence_list[i]])[0]))
    return np.array(sentence_embeddings)

def tsne(data, n_components=3):
    vectorizer = CountVectorizer()
    bow = vectorizer.fit_transform(data)
    if n_componants <= 3:
        return TSNE(n_components=n_components).fit_transform(bow)
    else:
        return TSNE(n_components=n_components, n_iter_without_progress=150, min_grad_norm=1e-5, verbose=2, method='exact', n_jobs=-1).fit_transform(bow)
