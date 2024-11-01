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
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=category_codes, cmap=color_map, marker='.')
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    # Create a custom legend
    unique_categories = np.unique(category_codes)
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(scatter.norm(code)), markersize=10) for code in unique_categories]
    # Set ticks inside the plot and make them thicker
    plt.tick_params(axis='both', direction='in', which='both', width=1)
    
    filename = title.lower().replace(' ', '_')
    plt.legend(legend_elements, category_labels, title="Research area", loc="upper right")
    plt.savefig(f"plots/rq2/{filename}.svg", format="svg")
    plt.savefig(f"plots/rq2/{filename}.png", format="png", dpi=300)
    plt.savefig(f"plots/rq2/{filename}.pdf", format="pdf")


if True:
    titles_tfidf = compute_tfidf(titles)
    reduced_titles_tfids = reduce_dimensionality(titles_tfidf, method='tsne')
    plot_embeddings(reduced_titles_tfids, df['main_collection_area'], 'Reference Research Publication Titles — TF-IDF Embeddings — t-SNE', 'plasma')
    reduced_titles_tfids = reduce_dimensionality(titles_tfidf, method='pca')
    plot_embeddings(reduced_titles_tfids, df['main_collection_area'], 'Reference Research Publication Titles — TF-IDF Embeddings — PCA', 'plasma')

if True:
    sentence_embeddings_titles = compute_sentence_embeddings(titles)
    reduced_embeddings_titles = reduce_dimensionality(sentence_embeddings_titles, method='tsne')
    plot_embeddings(reduced_embeddings_titles, df['main_collection_area'], 'Reference Research Publication Titles — Sentence-BERT Embeddings — t-SNE', 'plasma')
    reduced_embeddings_titles = reduce_dimensionality(sentence_embeddings_titles, method='pca')
    plot_embeddings(reduced_embeddings_titles, df['main_collection_area'], 'Reference Research Publication Titles — Sentence-BERT Embeddings — PCA', 'plasma')

if True:
    clip_embeddings_titles = compute_clip_embeddings(titles)
    reduced_clip_titles = reduce_dimensionality(clip_embeddings_titles, method='tsne')
    plot_embeddings(reduced_clip_titles, df['main_collection_area'], 'Reference Research Publication Titles — CLIP Embeddings — t-SNE', 'plasma')
    reduced_clip_titles = reduce_dimensionality(clip_embeddings_titles, method='pca')
    plot_embeddings(reduced_clip_titles, df['main_collection_area'], 'Reference Research Publication Titles — CLIP Embeddings — PCA', 'plasma')

# Abstract
if True:
    abstracts_tfidf = compute_tfidf(abstracts)
    reduced_abstracts_tfidf = reduce_dimensionality(abstracts_tfidf, method='tsne')
    plot_embeddings(reduced_abstracts_tfidf, df['main_collection_area'], 'Reference Research Publication Abstracts — TF-IDF Embeddings — t-SNE', 'plasma')
    reduced_abstracts_tfidf = reduce_dimensionality(abstracts_tfidf, method='pca')
    plot_embeddings(reduced_abstracts_tfidf, df['main_collection_area'], 'Reference Research Publication Abstracts — TF-IDF Embeddings — PCA', 'plasma')

if True:
    sentence_embeddings_abstracts = compute_sentence_embeddings(abstracts)
    reduced_embeddings_abstracts = reduce_dimensionality(sentence_embeddings_abstracts, method='tsne')
    plot_embeddings(reduced_embeddings_abstracts, df['main_collection_area'], 'Reference Research Publication Abstracts — Sentence-BERT Embeddings — t-SNE', 'plasma')
    reduced_embeddings_abstracts = reduce_dimensionality(sentence_embeddings_abstracts, method='pca')
    plot_embeddings(reduced_embeddings_abstracts, df['main_collection_area'], 'Reference Research Publication Abstracts — Sentence-BERT Embeddings — PCA', 'plasma')

if True:
    clip_embeddings_abstracts = compute_clip_embeddings(abstracts)
    reduced_clip_abstracts = reduce_dimensionality(clip_embeddings_abstracts, method='tsne')
    plot_embeddings(reduced_clip_abstracts, df['main_collection_area'], 'Reference Research Publication Abstracts — CLIP Embeddings — t-SNE', 'plasma')
    reduced_clip_abstracts = reduce_dimensionality(clip_embeddings_abstracts, method='pca')
    plot_embeddings(reduced_clip_abstracts, df['main_collection_area'], 'Reference Research Publication Abstracts — CLIP Embeddings — PCA', 'plasma')

# Repository READMEs
if True:
    readmes_tfidf = compute_tfidf(readmes)
    reduced_readmes_tfids = reduce_dimensionality(readmes_tfidf, method='tsne')
    plot_embeddings(reduced_readmes_tfids, df['main_collection_area'], 'Repository READMEs — TF-IDF Embeddings — t-SNE', 'plasma')
    reduced_readmes_tfids = reduce_dimensionality(readmes_tfidf, method='pca')
    plot_embeddings(reduced_readmes_tfids, df['main_collection_area'], 'Repository READMEs — TF-IDF Embeddings — PCA', 'plasma')

if True:
    sentence_embeddings_readmes = compute_sentence_embeddings(readmes)
    reduced_embeddings_readmes = reduce_dimensionality(sentence_embeddings_readmes, method='tsne')
    plot_embeddings(reduced_embeddings_readmes, df['main_collection_area'], 'Repository READMEs — Sentence-BERT Embeddings — t-SNE', 'plasma')
    reduced_embeddings_readmes = reduce_dimensionality(sentence_embeddings_readmes, method='pca')
    plot_embeddings(reduced_embeddings_readmes, df['main_collection_area'], 'Repository READMEs — Sentence-BERT Embeddings — PCA', 'plasma')

if True:
    clip_embeddings_readmes = compute_clip_embeddings(readmes)
    reduced_clip_readmes = reduce_dimensionality(clip_embeddings_readmes, method='tsne')
    plot_embeddings(reduced_clip_readmes, df['main_collection_area'], 'Repository READMEs — CLIP Embeddings — t-SNE', 'plasma')
    reduced_clip_readmes = reduce_dimensionality(clip_embeddings_readmes, method='pca')
    plot_embeddings(reduced_clip_readmes, df['main_collection_area'], 'Repository READMEs — CLIP Embeddings — PCA', 'plasma')

# Repository
if True:
    somef_tfidf = compute_tfidf(somef)
    reduced_somef_tfids = reduce_dimensionality(somef_tfidf, method='tsne')
    plot_embeddings(reduced_somef_tfids, df_somef['main_collection_area'], 'Repository Descriptions — TF-IDF Embeddings — t-SNE', 'plasma')
    reduced_somef_tfids = reduce_dimensionality(somef_tfidf, method='pca')
    plot_embeddings(reduced_somef_tfids, df_somef['main_collection_area'], 'Repository Descriptions — TF-IDF Embeddings — PCA', 'plasma')

if True:
    sentence_embeddings_somef = compute_sentence_embeddings(somef)
    reduced_embeddings_somef = reduce_dimensionality(sentence_embeddings_somef, method='tsne')
    plot_embeddings(reduced_embeddings_somef, df_somef['main_collection_area'], 'Repository Descriptions — Sentence-BERT Embeddings — t-SNE', 'plasma')
    reduced_embeddings_somef = reduce_dimensionality(sentence_embeddings_somef, method='pca')
    plot_embeddings(reduced_embeddings_somef, df_somef['main_collection_area'], 'Repository Descriptions — Sentence-BERT Embeddings — PCA', 'plasma')

if True:
    clip_embeddings_somef = compute_clip_embeddings(somef)
    reduced_clip_somef = reduce_dimensionality(clip_embeddings_somef, method='tsne')
    plot_embeddings(reduced_clip_somef, df_somef['main_collection_area'], 'Repository Descriptions — CLIP Embeddings — t-SNE', 'plasma')
    reduced_clip_somef = reduce_dimensionality(clip_embeddings_somef, method='pca')
    plot_embeddings(reduced_clip_somef, df_somef['main_collection_area'], 'Repository Descriptions — CLIP Embeddings — PCA', 'plasma')

# Repository Titles
if True:
    github_titles_tfidf = compute_tfidf(github_title)
    reduced_github_titles_tfids = reduce_dimensionality(github_titles_tfidf, method='tsne')
    plot_embeddings(reduced_github_titles_tfids, df_complete['main_collection_area'], 'Repository Titles — TF-IDF Embeddings — t-SNE', 'plasma')
    reduced_github_titles_tfids = reduce_dimensionality(github_titles_tfidf, method='pca')
    plot_embeddings(reduced_github_titles_tfids, df_complete['main_collection_area'], 'Repository Titles — TF-IDF Embeddings — PCA', 'plasma')

if True:
    sentence_embeddings_github_titles = compute_sentence_embeddings(github_title)
    reduced_embeddings_github_titles = reduce_dimensionality(sentence_embeddings_github_titles, method='tsne')
    plot_embeddings(reduced_embeddings_github_titles, df_complete['main_collection_area'], 'Repository Titles — Sentence-BERT Embeddings — t-SNE', 'plasma')
    reduced_embeddings_github_titles = reduce_dimensionality(sentence_embeddings_github_titles, method='pca')
    plot_embeddings(reduced_embeddings_github_titles, df_complete['main_collection_area'], 'Repository Titles — Sentence-BERT Embeddings — PCA', 'plasma')

if True:
    clip_embeddings_github_titles = compute_clip_embeddings(github_title)
    reduced_clip_github_titles = reduce_dimensionality(clip_embeddings_github_titles, method='tsne')
    plot_embeddings(reduced_clip_github_titles, df_complete['main_collection_area'], 'Repository Titles — CLIP Embeddings — t-SNE', 'plasma')
    reduced_clip_github_titles = reduce_dimensionality(clip_embeddings_github_titles, method='pca')
    plot_embeddings(reduced_clip_github_titles, df_complete['main_collection_area'], 'Repository Titles — CLIP Embeddings — PCA', 'plasma')

# Repository Keywords
if True:
    github_keywords_tfidf = compute_tfidf(github_keywords)
    reduced_github_keywords_tfids = reduce_dimensionality(github_keywords_tfidf, method='tsne')
    plot_embeddings(reduced_github_keywords_tfids, df_complete['main_collection_area'], 'Repository Keywords — TF-IDF Embeddings — t-SNE', 'plasma')
    reduced_github_keywords_tfids = reduce_dimensionality(github_keywords_tfidf, method='pca')
    plot_embeddings(reduced_github_keywords_tfids, df_complete['main_collection_area'], 'Repository Keywords — TF-IDF Embeddings — PCA', 'plasma')

if True:
    sentence_embeddings_github_keywords = compute_sentence_embeddings(github_keywords)
    reduced_embeddings_github_keywords = reduce_dimensionality(sentence_embeddings_github_keywords, method='tsne')
    plot_embeddings(reduced_embeddings_github_keywords, df_complete['main_collection_area'], 'Repository Keywords — Sentence-BERT Embeddings — t-SNE', 'plasma')
    reduced_embeddings_github_keywords = reduce_dimensionality(sentence_embeddings_github_keywords, method='pca')
    plot_embeddings(reduced_embeddings_github_keywords, df_complete['main_collection_area'], 'Repository Keywords — Sentence-BERT Embeddings — PCA', 'plasma')

if True:
    clip_embeddings_github_keywords = compute_clip_embeddings(github_keywords)
    reduced_clip_github_keywords = reduce_dimensionality(clip_embeddings_github_keywords, method='tsne')
    plot_embeddings(reduced_clip_github_keywords, df_complete['main_collection_area'], 'Repository Keywords — CLIP Embeddings — t-SNE', 'plasma')
    reduced_clip_github_keywords = reduce_dimensionality(clip_embeddings_github_keywords, method='pca')
    plot_embeddings(reduced_clip_github_keywords, df_complete['main_collection_area'], 'Repository Keywords — CLIP Embeddings — PCA', 'plasma')
