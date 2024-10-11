import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import CLIPTokenizer, CLIPModel
from sentence_transformers import SentenceTransformer
import Levenshtein
import numpy as np
import torch
import re
import string
from nltk.corpus import stopwords


model_bert = SentenceTransformer('all-MiniLM-L6-v2')
model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")


print('Load data')
with open('data/filtered_dataset.json', 'r') as f:
    papers_data = json.load(f)

df = pd.DataFrame(papers_data)
print(f'Number of samples: {df.shape[1]}')

titles = df['paper_title'].tolist()
abstracts = df['abstract'].tolist()
readmes = df['github_readme_content'].tolist()
somef = df['somef_descriptions'].tolist()

print('Load data')
with open('data/filtered_data_complete.json', 'r') as f:
    papers_data_complete = json.load(f)
df_complete = pd.DataFrame(papers_data_complete)
print(f'Number of samples: {df.shape[1]}')
github_title = df_complete['github_repo_title'].tolist()
github_keywords = df_complete['github_keywords'].tolist()


def preprocess_text(text):
    # Remove punctuation, numbers, and lower the text
    text = re.sub(f"[{string.punctuation}0-9]", " ", text.lower())
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text


def compute_tfidf(text_list):
    # Preprocess each text in the list
    text_list = [preprocess_text(text) for text in text_list]
    
    # Define the TF-IDF vectorizer with tuning
    vectorizer = TfidfVectorizer(min_df=2, max_df=0.95, ngram_range=(1, 2), stop_words='english', max_features=3000)
    
    vectors = vectorizer.fit_transform(text_list)
    return vectors.toarray()

def compute_sentence_embeddings(text_list, batch_size=256):
    embeddings = []
    text_list = [text.strip() for text in text_list]

    for i in range(0, len(text_list), batch_size):
        print(i)
        batch = text_list[i:i + batch_size]
        batch_embeddings = model_bert.encode(batch)
        embeddings.append(batch_embeddings)

    # Concatenate all batch embeddings
    return np.vstack(embeddings)


def compute_clip_embeddings(text_list, batch_size=256):
    embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            batch_embeddings = model_clip.get_text_features(**inputs).cpu().numpy()
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)


def reduce_dimensionality(embeddings, method):
    if method == 'pca':
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeddings)
    elif method == 'tsne':
        tsne = TSNE(n_components=2, random_state=42)
        reduced = tsne.fit_transform(embeddings)
    return reduced


def plot_embeddings(embeddings, color_by, title, color_map):
    plt.figure(figsize=(15, 15))

    # Convert color_by into a categorical type and get unique categories
    categories = pd.Categorical(color_by)
    category_codes = categories.codes
    category_labels = categories.categories
    
    # Create scatter plot
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=category_codes, cmap=color_map)

    plt.title(title)
    
    # Create a custom legend
    unique_categories = np.unique(category_codes)
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(scatter.norm(code)), markersize=10) 
                       for code in unique_categories]
    plt.legend(legend_elements, category_labels, title="Category", loc="upper right")

    filename = title.replace(' ', '_')
    plt.savefig(f'plots/{filename}.png')

# Title
if True:
    titles_tfidf = compute_tfidf(titles)
    reduced_titles_tfids = reduce_dimensionality(titles_tfidf, method='tsne')
    plot_embeddings(reduced_titles_tfids, df['main_collection_area'], f'TF-IDF Embeddings (Titles #{len(titles)} ) - TSNE - Colored by Area', 'plasma')
    reduced_titles_tfids = reduce_dimensionality(titles_tfidf, method='pca')
    plot_embeddings(reduced_titles_tfids, df['main_collection_area'], f'TF-IDF Embeddings (Titles #{len(titles)} ) - PCA - Colored by Area', 'plasma')

if True:
    sentence_embeddings_titles = compute_sentence_embeddings(titles)
    reduced_embeddings_titles = reduce_dimensionality(sentence_embeddings_titles, method='tsne')
    plot_embeddings(reduced_embeddings_titles, df['main_collection_area'], f'Sentence-BERT Embeddings (Titles #{len(titles)} ) - TSNE - Colored by Area', 'plasma')
    reduced_embeddings_titles = reduce_dimensionality(sentence_embeddings_titles, method='pca')
    plot_embeddings(reduced_embeddings_titles, df['main_collection_area'], f'Sentence-BERT Embeddings (Titles #{len(titles)} ) - PCA - Colored by Area', 'plasma')

if True:
    clip_embeddings_titles = compute_clip_embeddings(titles)
    reduced_clip_titles = reduce_dimensionality(clip_embeddings_titles, method='tsne')
    plot_embeddings(reduced_clip_titles, df['main_collection_area'], f'CLIP Embeddings (Titles #{len(titles)} ) - TSNE - Colored by Area', 'plasma')
    reduced_clip_titles = reduce_dimensionality(clip_embeddings_titles, method='pca')
    plot_embeddings(reduced_clip_titles, df['main_collection_area'], f'CLIP Embeddings (Titles #{len(titles)} ) - PCA - Colored by Area', 'plasma')

# Abstract
if True:
    abstracts_tfidf = compute_tfidf(abstracts)
    reduced_abstracts_tfidf = reduce_dimensionality(abstracts_tfidf, method='tsne')
    plot_embeddings(reduced_abstracts_tfidf, df['main_collection_area'], f'TF-IDF Embeddings (Abstracts #{len(abstracts)} ) - TSNE - Colored by Area', 'plasma')
    reduced_abstracts_tfidf = reduce_dimensionality(abstracts_tfidf, method='pca')
    plot_embeddings(reduced_abstracts_tfidf, df['main_collection_area'], f'TF-IDF Embeddings (Abstracts #{len(abstracts)} ) - PCA - Colored by Area', 'plasma')

if True:
    sentence_embeddings_abstracts = compute_sentence_embeddings(abstracts)
    reduced_embeddings_abstracts = reduce_dimensionality(sentence_embeddings_abstracts, method='tsne')
    plot_embeddings(reduced_embeddings_abstracts, df['main_collection_area'], f'Sentence-BERT Embeddings (Abstracts #{len(abstracts)} ) - TSNE - Colored by Area', 'plasma')
    reduced_embeddings_abstracts = reduce_dimensionality(sentence_embeddings_abstracts, method='pca')
    plot_embeddings(reduced_embeddings_abstracts, df['main_collection_area'], f'Sentence-BERT Embeddings (Abstracts #{len(abstracts)} ) - PCA - Colored by Area', 'plasma')

if True:
    clip_embeddings_abstracts = compute_clip_embeddings(abstracts)
    reduced_clip_abstracts = reduce_dimensionality(clip_embeddings_abstracts, method='tsne')
    plot_embeddings(reduced_clip_abstracts, df['main_collection_area'], f'CLIP Embeddings (Abstracts #{len(abstracts)} ) - TSNE - Colored by Area', 'plasma')
    reduced_clip_abstracts = reduce_dimensionality(clip_embeddings_abstracts, method='pca')
    plot_embeddings(reduced_clip_abstracts, df['main_collection_area'], f'CLIP Embeddings (Abstracts #{len(abstracts)} ) - PCA - Colored by Area', 'plasma')

# Readme
if True:
    readmes_tfidf = compute_tfidf(readmes)
    reduced_readmes_tfids = reduce_dimensionality(readmes_tfidf, method='tsne')
    plot_embeddings(reduced_readmes_tfids, df['main_collection_area'], f'TF-IDF Embeddings (READMEs #{len(readmes)} ) - TSNE - Colored by Area', 'plasma')
    reduced_readmes_tfids = reduce_dimensionality(readmes_tfidf, method='pca')
    plot_embeddings(reduced_readmes_tfids, df['main_collection_area'], f'TF-IDF Embeddings (READMEs #{len(readmes)} ) - PCA - Colored by Area', 'plasma')

if True:
    sentence_embeddings_abstracts = compute_sentence_embeddings(readmes)
    reduced_embeddings_abstracts = reduce_dimensionality(sentence_embeddings_abstracts, method='tsne')
    plot_embeddings(reduced_embeddings_abstracts, df['main_collection_area'], f'Sentence-BERT Embeddings (READMEs #{len(readmes)} ) - TSNE - Colored by Area', 'plasma')
    reduced_embeddings_abstracts = reduce_dimensionality(sentence_embeddings_abstracts, method='pca')
    plot_embeddings(reduced_embeddings_abstracts, df['main_collection_area'], f'Sentence-BERT Embeddings (READMEs #{len(readmes)} ) - PCA - Colored by Area', 'plasma')

if True:
    clip_embeddings_readmes = compute_clip_embeddings(readmes)
    reduced_clip_readmes = reduce_dimensionality(clip_embeddings_readmes, method='tsne')
    plot_embeddings(reduced_clip_readmes, df['main_collection_area'], f'CLIP Embeddings (READMEs #{len(readmes)} ) - TSNE - Colored by Area', 'plasma')
    reduced_clip_readmes = reduce_dimensionality(clip_embeddings_readmes, method='pca')
    plot_embeddings(reduced_clip_readmes, df['main_collection_area'], f'CLIP Embeddings (READMEs #{len(readmes)} ) - PCA - Colored by Area', 'plasma')

# Somef
if True:
    readmes_tfidf = compute_tfidf(somef)
    reduced_readmes_tfids = reduce_dimensionality(readmes_tfidf, method='tsne')
    plot_embeddings(reduced_readmes_tfids, df['main_collection_area'], f'TF-IDF Embeddings (SOMEF descriptions #{len(somef)} ) - TSNE - Colored by Area', 'plasma')
    reduced_readmes_tfids = reduce_dimensionality(readmes_tfidf, method='pca')
    plot_embeddings(reduced_readmes_tfids, df['main_collection_area'], f'TF-IDF Embeddings (SOMEF descriptions #{len(somef)} ) - PCA - Colored by Area', 'plasma')

if True:
    sentence_embeddings_abstracts = compute_sentence_embeddings(somef)
    reduced_embeddings_abstracts = reduce_dimensionality(sentence_embeddings_abstracts, method='tsne')
    plot_embeddings(reduced_embeddings_abstracts, df['main_collection_area'], f'Sentence-BERT Embeddings (SOMEF descriptions #{len(somef)} ) - TSNE - Colored by Area', 'plasma')
    reduced_embeddings_abstracts = reduce_dimensionality(sentence_embeddings_abstracts, method='pca')
    plot_embeddings(reduced_embeddings_abstracts, df['main_collection_area'], f'Sentence-BERT Embeddings (SOMEF descriptions #{len(somef)} ) - PCA - Colored by Area', 'plasma')

if True:
    clip_embeddings_readmes = compute_clip_embeddings(somef)
    reduced_clip_readmes = reduce_dimensionality(clip_embeddings_readmes, method='tsne')
    plot_embeddings(reduced_clip_readmes, df['main_collection_area'], f'CLIP Embeddings (SOMEF descriptions #{len(somef)} ) - TSNE - Colored by Area', 'plasma')
    reduced_clip_readmes = reduce_dimensionality(clip_embeddings_readmes, method='pca')
    plot_embeddings(reduced_clip_readmes, df['main_collection_area'], f'CLIP Embeddings (SOMEF descriptions #{len(somef)} ) - PCA - Colored by Area', 'plasma')

# Github Titles
if True:
    readmes_tfidf = compute_tfidf(github_title)
    reduced_readmes_tfids = reduce_dimensionality(readmes_tfidf, method='tsne')
    plot_embeddings(reduced_readmes_tfids, df['main_collection_area'], f'TF-IDF Embeddings (GitHub Titles #{len(github_title)} ) - TSNE - Colored by Area', 'plasma')
    reduced_readmes_tfids = reduce_dimensionality(readmes_tfidf, method='pca')
    plot_embeddings(reduced_readmes_tfids, df['main_collection_area'], f'TF-IDF Embeddings (GitHub Titles #{len(github_title)} ) - PCA - Colored by Area', 'plasma')

if True:
    sentence_embeddings_abstracts = compute_sentence_embeddings(github_title)
    reduced_embeddings_abstracts = reduce_dimensionality(sentence_embeddings_abstracts, method='tsne')
    plot_embeddings(reduced_embeddings_abstracts, df['main_collection_area'], f'Sentence-BERT Embeddings (GitHub Titles #{len(github_title)} ) - TSNE - Colored by Area', 'plasma')
    reduced_embeddings_abstracts = reduce_dimensionality(sentence_embeddings_abstracts, method='pca')
    plot_embeddings(reduced_embeddings_abstracts, df['main_collection_area'], f'Sentence-BERT Embeddings (GitHub Titles #{len(github_title)} ) - PCA - Colored by Area', 'plasma')

if True:
    clip_embeddings_readmes = compute_clip_embeddings(github_title)
    reduced_clip_readmes = reduce_dimensionality(clip_embeddings_readmes, method='tsne')
    plot_embeddings(reduced_clip_readmes, df['main_collection_area'], f'CLIP Embeddings (GitHub Titles #{len(github_title)} ) - TSNE - Colored by Area', 'plasma')
    reduced_clip_readmes = reduce_dimensionality(clip_embeddings_readmes, method='pca')
    plot_embeddings(reduced_clip_readmes, df['main_collection_area'], f'CLIP Embeddings (GitHub Titles #{len(github_title)} ) - PCA - Colored by Area', 'plasma')

# Github Keywords
if True:
    readmes_tfidf = compute_tfidf(github_keywords)
    reduced_readmes_tfids = reduce_dimensionality(readmes_tfidf, method='tsne')
    plot_embeddings(reduced_readmes_tfids, df['main_collection_area'], f'TF-IDF Embeddings (GitHub Keywords #{len(github_keywords)} ) - TSNE - Colored by Area', 'plasma')
    reduced_readmes_tfids = reduce_dimensionality(readmes_tfidf, method='pca')
    plot_embeddings(reduced_readmes_tfids, df['main_collection_area'], f'TF-IDF Embeddings (GitHub Keywords #{len(github_keywords)} ) - PCA - Colored by Area', 'plasma')

if True:
    sentence_embeddings_abstracts = compute_sentence_embeddings(github_keywords)
    reduced_embeddings_abstracts = reduce_dimensionality(sentence_embeddings_abstracts, method='tsne')
    plot_embeddings(reduced_embeddings_abstracts, df['main_collection_area'], f'Sentence-BERT Embeddings (GitHub Keywords #{len(github_keywords)} ) - TSNE - Colored by Area', 'plasma')
    reduced_embeddings_abstracts = reduce_dimensionality(sentence_embeddings_abstracts, method='pca')
    plot_embeddings(reduced_embeddings_abstracts, df['main_collection_area'], f'Sentence-BERT Embeddings (GitHub Keywords #{len(github_keywords)} ) - PCA - Colored by Area', 'plasma')

if True:
    clip_embeddings_readmes = compute_clip_embeddings(github_keywords)
    reduced_clip_readmes = reduce_dimensionality(clip_embeddings_readmes, method='tsne')
    plot_embeddings(reduced_clip_readmes, df['main_collection_area'], f'CLIP Embeddings (GitHub Keywords #{len(github_keywords)} ) - TSNE - Colored by Area', 'plasma')
    reduced_clip_readmes = reduce_dimensionality(clip_embeddings_readmes, method='pca')
    plot_embeddings(reduced_clip_readmes, df['main_collection_area'], f'CLIP Embeddings (GitHub Keywords #{len(github_keywords)} ) - PCA - Colored by Area', 'plasma')

