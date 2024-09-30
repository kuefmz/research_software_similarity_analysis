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


def compute_tfidf(text_list, batch_size=32):
    vectorizer = TfidfVectorizer()

    # First, fit the vectorizer on the entire text_list to ensure consistent feature space
    vectorizer.fit(text_list)
    embeddings = []

    # Process in batches
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]
        # Use the pre-fitted vectorizer to transform the batch
        batch_vectors = vectorizer.transform(batch)
        batch_cosine_sim = cosine_similarity(batch_vectors)
        embeddings.append(batch_cosine_sim)

    # Concatenate all the batch results
    return np.vstack(embeddings)


def computer_tfidf_no_batches(text_list):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(text_list)
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim


def compute_sentence_embeddings(text_list, batch_size=256):
    model_bert = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = []

    # Process in batches

    for i in range(0, len(text_list), batch_size):
        print(i)
        batch = text_list[i:i + batch_size]
        batch_embeddings = model_bert.encode(batch)
        embeddings.append(batch_embeddings)

    # Concatenate all batch embeddings
    return np.vstack(embeddings)


def compute_clip_embeddings(text_list, batch_size=256):
    model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    embeddings = []

    # Process in batches
    for i in range(0, len(text_list), batch_size):
        print(i)
        batch = text_list[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            batch_embeddings = model_clip.get_text_features(**inputs).cpu().numpy()
        embeddings.append(batch_embeddings)

    # Concatenate all batch embeddings
    return np.vstack(embeddings)


def reduce_dimensionality(embeddings, method='pca'):
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


print('Load data')
with open('data/paper_title_abstract.json', 'r') as f:
    papers_data = json.load(f)

df = pd.DataFrame(papers_data)

titles = df['paper_title'].tolist()
abstracts = df['abstract'].tolist()


print('Compute TF-IDF')
if False:
    titles_tfidf = compute_tfidf(titles)
    reduced_titles_tfids = reduce_dimensionality(titles_tfidf)
    plot_embeddings(reduced_titles_tfids, df['main_collection_area'], 'TF-IDF Embeddings (Titles) - Colored by Area', 'plasma')

if False:
    abstracts_tfidf = compute_tfidf(abstracts)
    reduced_abstracts_tfidf = reduce_dimensionality(abstracts_tfidf)
    plot_embeddings(reduced_abstracts_tfidf, df['main_collection_area'], 'TF-IDF Embeddings (Abstracts) - Colored by Area', 'plasma')


print('BERT')
if False:
    sentence_embeddings_titles = compute_sentence_embeddings(titles)
    reduced_embeddings_titles = reduce_dimensionality(sentence_embeddings_titles)
    plot_embeddings(reduced_embeddings_titles, df['main_collection_area'], 'Sentence-BERT Embeddings (Titles) - Colored by Area', 'plasma')

if False:
    sentence_embeddings_abstracts = compute_sentence_embeddings(abstracts)
    reduced_embeddings_abstracts = reduce_dimensionality(sentence_embeddings_abstracts)
    plot_embeddings(reduced_embeddings_abstracts, df['main_collection_area'], 'Sentence-BERT Embeddings (Abstracts) - Colored by Area', 'plasma')


print('CLIP')
if False:
    clip_embeddings_titles = compute_clip_embeddings(titles)
    reduced_clip_titles = reduce_dimensionality(clip_embeddings_titles)
    plot_embeddings(reduced_clip_titles, df['main_collection_area'], 'CLIP Embeddings (Titles) - Colored by Area', 'plasma')

if False:
    clip_embeddings_abstracts = compute_clip_embeddings(abstracts)
    reduced_clip_abstracts = reduce_dimensionality(clip_embeddings_abstracts)
    plot_embeddings(reduced_clip_abstracts, df['main_collection_area'], 'CLIP Embeddings (Abstracts) - Colored by Area', 'plasma')


print('Load data')
with open('data/merged_papers_methods_with_github_readmes.json', 'r') as f:
    papers_data = json.load(f)

df = pd.DataFrame(papers_data)

print(df.shape)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
print(df.shape)

print(df['main_collection_area'].value_counts())

readmes = df['github_readme_content'].tolist()

if True:
    readmes_tfidf = computer_tfidf_no_batches(readmes)
    reduced_readmes_tfids = reduce_dimensionality(readmes_tfidf)
    plot_embeddings(reduced_readmes_tfids, df['main_collection_area'], 'TF-IDF Embeddings (READMEs) - Colored by Area', 'plasma')

if True:
    sentence_embeddings_abstracts = compute_sentence_embeddings(readmes)
    reduced_embeddings_abstracts = reduce_dimensionality(sentence_embeddings_abstracts)
    plot_embeddings(reduced_embeddings_abstracts, df['main_collection_area'], 'Sentence-BERT Embeddings (READMEs) - Colored by Area', 'plasma')

if True:
    clip_embeddings_readmes = compute_clip_embeddings(readmes)
    reduced_clip_readmes = reduce_dimensionality(clip_embeddings_readmes)
    plot_embeddings(reduced_clip_readmes, df['main_collection_area'], 'CLIP Embeddings (READMEs) - Colored by Area', 'plasma')
