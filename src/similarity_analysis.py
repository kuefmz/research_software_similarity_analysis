import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
from utils.vectorizers import compute_tfidf, compute_sentence_embeddings, compute_clip_embeddings 

print('Load data')
with open('data/final_dataset.json', 'r') as f:
    papers_data = json.load(f)

df = pd.DataFrame(papers_data)
print(f'Number of samples: {df.shape[0]}')

titles = df['paper_title'].tolist()
abstracts = df['abstract'].tolist()
readmes = df['github_readme_content'].tolist()


print('Load data for somef decriptions')
with open('data/filtered_data.json', 'r') as f:
    papers_data_somef = json.load(f)

df_somef = pd.DataFrame(papers_data_somef)
print(f'Number of samples: {df_somef.shape[0]}')

somef = df_somef['somef_descriptions'].tolist()

print('Load data for github titles and keywords')
with open('data/filtered_data_complete.json', 'r') as f:
    papers_data_complete = json.load(f)

df_complete = pd.DataFrame(papers_data_complete)
print(f'Number of samples: {df_complete.shape[0]}')
github_title = df_complete['github_repo_title'].tolist()
github_keywords = df_complete['github_keywords'].tolist()


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


if False:
    titles_tfidf = compute_tfidf(titles)
    reduced_titles_tfids = reduce_dimensionality(titles_tfidf, method='tsne')
    plot_embeddings_plotly(reduced_titles_tfids, df['main_collection_area'], 'TF-IDF Embeddings (T-SNE) - Paper Titles', 'plasma')
    reduced_titles_tfids = reduce_dimensionality(titles_tfidf, method='pca')
    plot_embeddings_plotly(reduced_titles_tfids, df['main_collection_area'], 'TF-IDF Embeddings (PCA) - Paper Titles', 'plasma')

if False:
    sentence_embeddings_titles = compute_sentence_embeddings(titles)
    reduced_embeddings_titles = reduce_dimensionality(sentence_embeddings_titles, method='tsne')
    plot_embeddings(reduced_embeddings_titles, df['main_collection_area'], 'Sentence-BERT Embeddings (T-SNE) - Paper Titles', 'plasma')
    reduced_embeddings_titles = reduce_dimensionality(sentence_embeddings_titles, method='pca')
    plot_embeddings(reduced_embeddings_titles, df['main_collection_area'], 'Sentence-BERT Embeddings (PCA) - Paper Titles', 'plasma')

if False:
    clip_embeddings_titles = compute_clip_embeddings(titles)
    reduced_clip_titles = reduce_dimensionality(clip_embeddings_titles, method='tsne')
    plot_embeddings(reduced_clip_titles, df['main_collection_area'], 'CLIP Embeddings (T-SNE) - Paper Titles', 'plasma')
    reduced_clip_titles = reduce_dimensionality(clip_embeddings_titles, method='pca')
    plot_embeddings(reduced_clip_titles, df['main_collection_area'], 'CLIP Embeddings (PCA) - Paper Titles', 'plasma')

# Abstract
if False:
    abstracts_tfidf = compute_tfidf(abstracts)
    reduced_abstracts_tfidf = reduce_dimensionality(abstracts_tfidf, method='tsne')
    plot_embeddings(reduced_abstracts_tfidf, df['main_collection_area'], 'TF-IDF Embeddings (T-SNE) - Abstracts', 'plasma')
    reduced_abstracts_tfidf = reduce_dimensionality(abstracts_tfidf, method='pca')
    plot_embeddings(reduced_abstracts_tfidf, df['main_collection_area'], 'TF-IDF Embeddings (PCA) - Abstracts', 'plasma')

if False:
    sentence_embeddings_abstracts = compute_sentence_embeddings(abstracts)
    reduced_embeddings_abstracts = reduce_dimensionality(sentence_embeddings_abstracts, method='tsne')
    plot_embeddings(reduced_embeddings_abstracts, df['main_collection_area'], 'Sentence-BERT Embeddings (T-SNE) - Abstracts', 'plasma')
    reduced_embeddings_abstracts = reduce_dimensionality(sentence_embeddings_abstracts, method='pca')
    plot_embeddings(reduced_embeddings_abstracts, df['main_collection_area'], 'Sentence-BERT Embeddings (PCA) - Abstracts', 'plasma')

if False:
    clip_embeddings_abstracts = compute_clip_embeddings(abstracts)
    reduced_clip_abstracts = reduce_dimensionality(clip_embeddings_abstracts, method='tsne')
    plot_embeddings(reduced_clip_abstracts, df['main_collection_area'], 'CLIP Embeddings (T-SNE) - Abstracts', 'plasma')
    reduced_clip_abstracts = reduce_dimensionality(clip_embeddings_abstracts, method='pca')
    plot_embeddings(reduced_clip_abstracts, df['main_collection_area'], 'CLIP Embeddings (PCA) - Abstracts', 'plasma')

# Readme
if False:
    readmes_tfidf = compute_tfidf(readmes)
    reduced_readmes_tfids = reduce_dimensionality(readmes_tfidf, method='tsne')
    plot_embeddings(reduced_readmes_tfids, df['main_collection_area'], 'TF-IDF Embeddings (T-SNE) - READMEs', 'plasma')
    reduced_readmes_tfids = reduce_dimensionality(readmes_tfidf, method='pca')
    plot_embeddings(reduced_readmes_tfids, df['main_collection_area'], 'TF-IDF Embeddings (PCA) - READMEs', 'plasma')

if False:
    sentence_embeddings_readmes = compute_sentence_embeddings(readmes)
    reduced_embeddings_readmes = reduce_dimensionality(sentence_embeddings_readmes, method='tsne')
    plot_embeddings(reduced_embeddings_readmes, df['main_collection_area'], 'Sentence-BERT Embeddings (T-SNE) - READMEs', 'plasma')
    reduced_embeddings_readmes = reduce_dimensionality(sentence_embeddings_readmes, method='pca')
    plot_embeddings(reduced_embeddings_readmes, df['main_collection_area'], 'Sentence-BERT Embeddings (PCA) - READMEs', 'plasma')

if False:
    clip_embeddings_readmes = compute_clip_embeddings(readmes)
    reduced_clip_readmes = reduce_dimensionality(clip_embeddings_readmes, method='tsne')
    plot_embeddings(reduced_clip_readmes, df['main_collection_area'], 'CLIP Embeddings (T-SNE) - READMEs', 'plasma')
    reduced_clip_readmes = reduce_dimensionality(clip_embeddings_readmes, method='pca')
    plot_embeddings(reduced_clip_readmes, df['main_collection_area'], 'CLIP Embeddings (PCA) - READMEs', 'plasma')

# Somef
if False:
    somef_tfidf = compute_tfidf(somef)
    reduced_somef_tfids = reduce_dimensionality(somef_tfidf, method='tsne')
    plot_embeddings(reduced_somef_tfids, df_somef['main_collection_area'], 'TF-IDF Embeddings (T-SNE) - SOMEF Descriptions', 'plasma')
    reduced_somef_tfids = reduce_dimensionality(somef_tfidf, method='pca')
    plot_embeddings(reduced_somef_tfids, df_somef['main_collection_area'], 'TF-IDF Embeddings (PCA) - SOMEF Descriptions', 'plasma')

if False:
    sentence_embeddings_somef = compute_sentence_embeddings(somef)
    reduced_embeddings_somef = reduce_dimensionality(sentence_embeddings_somef, method='tsne')
    plot_embeddings(reduced_embeddings_somef, df_somef['main_collection_area'], 'Sentence-BERT Embeddings (T-SNE) - SOMEF Descriptions', 'plasma')
    reduced_embeddings_somef = reduce_dimensionality(sentence_embeddings_somef, method='pca')
    plot_embeddings(reduced_embeddings_somef, df_somef['main_collection_area'], 'Sentence-BERT Embeddings (PCA) - SOMEF Descriptions', 'plasma')

if False:
    clip_embeddings_somef = compute_clip_embeddings(somef)
    reduced_clip_somef = reduce_dimensionality(clip_embeddings_somef, method='tsne')
    plot_embeddings(reduced_clip_somef, df_somef['main_collection_area'], 'CLIP Embeddings (T-SNE) - SOMEF Descriptions', 'plasma')
    reduced_clip_somef = reduce_dimensionality(clip_embeddings_somef, method='pca')
    plot_embeddings(reduced_clip_somef, df_somef['main_collection_area'], 'CLIP Embeddings (PCA) - SOMEF Descriptions', 'plasma')

# Github Titles
if False:
    github_titles_tfidf = compute_tfidf(github_title)
    reduced_github_titles_tfids = reduce_dimensionality(github_titles_tfidf, method='tsne')
    plot_embeddings(reduced_github_titles_tfids, df_complete['main_collection_area'], 'TF-IDF Embeddings (T-SNE) - GitHub Titles', 'plasma')
    reduced_github_titles_tfids = reduce_dimensionality(github_titles_tfidf, method='pca')
    plot_embeddings(reduced_github_titles_tfids, df_complete['main_collection_area'], 'TF-IDF Embeddings (PCA) - GitHub Titles', 'plasma')

if False:
    sentence_embeddings_github_titles = compute_sentence_embeddings(github_title)
    reduced_embeddings_github_titles = reduce_dimensionality(sentence_embeddings_github_titles, method='tsne')
    plot_embeddings(reduced_embeddings_github_titles, df_complete['main_collection_area'], 'Sentence-BERT Embeddings (T-SNE) - GitHub Titles', 'plasma')
    reduced_embeddings_github_titles = reduce_dimensionality(sentence_embeddings_github_titles, method='pca')
    plot_embeddings(reduced_embeddings_github_titles, df_complete['main_collection_area'], 'Sentence-BERT Embeddings (PCA) - GitHub Titles', 'plasma')

if False:
    clip_embeddings_github_titles = compute_clip_embeddings(github_title)
    reduced_clip_github_titles = reduce_dimensionality(clip_embeddings_github_titles, method='tsne')
    plot_embeddings(reduced_clip_github_titles, df_complete['main_collection_area'], 'CLIP Embeddings (T-SNE) - GitHub Titles', 'plasma')
    reduced_clip_github_titles = reduce_dimensionality(clip_embeddings_github_titles, method='pca')
    plot_embeddings(reduced_clip_github_titles, df_complete['main_collection_area'], 'CLIP Embeddings (PCA) - GitHub Titles', 'plasma')

# Github Keywords
if True:
    github_keywords_tfidf = compute_tfidf(github_keywords)
    reduced_github_keywords_tfids = reduce_dimensionality(github_keywords_tfidf, method='tsne')
    plot_embeddings(reduced_github_keywords_tfids, df_complete['main_collection_area'], 'TF-IDF Embeddings (T-SNE) - GitHub Keywords', 'plasma')
    reduced_github_keywords_tfids = reduce_dimensionality(github_keywords_tfidf, method='pca')
    plot_embeddings(reduced_github_keywords_tfids, df_complete['main_collection_area'], 'TF-IDF Embeddings (PCA) - GitHub Keywords', 'plasma')

if False:
    sentence_embeddings_github_keywords = compute_sentence_embeddings(github_keywords)
    reduced_embeddings_github_keywords = reduce_dimensionality(sentence_embeddings_github_keywords, method='tsne')
    plot_embeddings(reduced_embeddings_github_keywords, df_complete['main_collection_area'], 'Sentence-BERT Embeddings (T-SNE) - GitHub Keywords', 'plasma')
    reduced_embeddings_github_keywords = reduce_dimensionality(sentence_embeddings_github_keywords, method='pca')
    plot_embeddings(reduced_embeddings_github_keywords, df_complete['main_collection_area'], 'Sentence-BERT Embeddings (PCA) - GitHub Keywords', 'plasma')

if False:
    clip_embeddings_github_keywords = compute_clip_embeddings(github_keywords)
    reduced_clip_github_keywords = reduce_dimensionality(clip_embeddings_github_keywords, method='tsne')
    plot_embeddings(reduced_clip_github_keywords, df_complete['main_collection_area'], 'CLIP Embeddings (T-SNE) - GitHub Keywords', 'plasma')
    reduced_clip_github_keywords = reduce_dimensionality(clip_embeddings_github_keywords, method='pca')
    plot_embeddings(reduced_clip_github_keywords, df_complete['main_collection_area'], 'CLIP Embeddings (PCA) - GitHub Keywords', 'plasma')
