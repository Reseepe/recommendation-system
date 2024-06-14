import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json


def load_bert_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def compute_bert_embeddings(model, text_data):
    return model.encode(text_data, convert_to_tensor=True)


def load_embeddings(file_name):
    with open(file_name, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings

def load_vectorizer(file_name):
    with open(file_name, 'r') as f:
        vectorizer_data = json.load(f)
    
    vectorizer = TfidfVectorizer()
    vectorizer.vocabulary_ = vectorizer_data['vocabulary_']
    vectorizer.idf_ = np.array(vectorizer_data['idf_'])
    vectorizer.stop_words_ = set(vectorizer_data['stop_words_'])
    return vectorizer

def find_similar_recipes(user_input):
    # Load pre-computed embeddings
    data = pd.read_pickle('app/food_sample.pkl')
    data['text_data'] = data[['name', 'tags', 'description']].astype(str).agg(' '.join, axis=1)

    # Load pre-computed embeddings
    bert_embeddings = load_embeddings('app/sample_bert_embeddings.pkl')
    tfidf_embeddings = load_embeddings('app/sample_tfidf_embeddings.pkl')
    vectorizer = load_vectorizer('app/sample_tfidf_vectorizer.json')

    # Encode user input
    bert_model = load_bert_model()
    user_bert_embedding = compute_bert_embeddings(bert_model, [user_input])
    user_tfidf_embedding = vectorizer.transform([user_input])

    # Compute cosine similarity with BERT embeddings
    bert_cosine_sim_matrix = util.pytorch_cos_sim(user_bert_embedding, bert_embeddings).numpy()
    bert_similar_recipes = bert_cosine_sim_matrix[0].argsort()[::-1][:20]

    # Compute cosine similarity with TF-IDF embeddings
    tfidf_cosine_sim_matrix = cosine_similarity(user_tfidf_embedding, tfidf_embeddings)
    tfidf_similar_recipes = tfidf_cosine_sim_matrix[0].argsort()[::-1][:20]

    # Combine results from both embeddings
    combined_similar_recipes = np.union1d(bert_similar_recipes, tfidf_similar_recipes)

    # Get similar recipe names from sampled_data
    similar_recipe_names = data.iloc[combined_similar_recipes]['name'].tolist()
    return similar_recipe_names
