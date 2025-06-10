import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import euclidean_distances
from sentence_transformers import SentenceTransformer
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from adjustText import adjust_text
import os


# ====== CONFIG =======
DATASET_PATH = "Data/sample_set_with_language.csv"
N_SAMPLES = 50000
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
N_CLUSTERS = 20
SAVE_PATH = "Data/clustered_sample_set.csv"
# ======================

def embed_lyrics(df, column='lyrics', model_name=EMBEDDING_MODEL):
    model = SentenceTransformer(model_name)
    df = df[df[column].notnull() & df[column].apply(lambda x: isinstance(x, str) and len(x) > 30)]
    embeddings = model.encode(df[column].tolist(), show_progress_bar=True)
    df['lyrics_embedding'] = list(embeddings)
    return df

def cluster_songs(features, n_clusters=N_CLUSTERS):
    reducer = PCA(n_components=20)
    reduced = reducer.fit_transform(features)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(reduced)
    return clusters, reduced

def get_representative_songs(reduced_features, labels, df):
    representatives = []
    for cluster_id in sorted(set(labels)):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_points = reduced_features[cluster_indices]
        centroid = np.mean(cluster_points, axis=0).reshape(1, -1)
        distances = euclidean_distances(cluster_points, centroid).flatten()
        closest_index = cluster_indices[np.argmin(distances)]
        representatives.append(df.iloc[closest_index])
    return pd.DataFrame(representatives)

def visualize(reduced_features, labels, df):
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
    embedding = tsne.fit_transform(reduced_features)

    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab20', s=50, alpha=0.7)

    texts = []
    for i, (x, y) in enumerate(embedding):
        if random.random() < 0.01:
            label = df.iloc[i]['title'] if 'title' in df.columns else 'Song'
            texts.append(plt.text(x, y, label[:20], fontsize=8, alpha=0.8))
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    handles = [plt.Line2D([], [], marker='o', color='w',
                          label=f'Cluster {i}', markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=10)
               for i in range(len(set(labels)))]
    plt.legend(handles=handles, title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title('Song Clusters via t-SNE')
    plt.tight_layout()
    plt.show()

def main():
    print("Current working directory:", os.getcwd())
    
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)

    print("Embedding lyrics...")
    df = embed_lyrics(df)

    print("Clustering...")
    features = np.vstack(df['lyrics_embedding'])
    labels, reduced = cluster_songs(features)
    df['cluster'] = labels

    print("Saving clustered dataset...")
    df.to_csv(SAVE_PATH, index=False)

    print("Getting representative songs...")
    reps = get_representative_songs(reduced, labels, df)
    print(reps[['title', 'artist', 'cluster']].head())

    print("Visualizing...")
    visualize(reduced, labels, df)

if __name__ == "__main__":
    main()
