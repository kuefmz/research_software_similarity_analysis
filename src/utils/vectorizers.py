from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import CLIPTokenizer, CLIPModel
from sentence_transformers import SentenceTransformer
import torch
import numpy as np


model_bert = SentenceTransformer('all-MiniLM-L6-v2')
model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")


def compute_tfidf(text_list):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000, sublinear_tf=True)
    vectors = vectorizer.fit_transform(text_list)
    return vectors.toarray()


def compute_sentence_embeddings(text_list, batch_size=256):
    embeddings = []
    text_list = [text.strip() for text in text_list]

    for i in range(0, len(text_list), batch_size):
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