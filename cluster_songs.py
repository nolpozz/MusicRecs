import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sentence_transformers import SentenceTransformer
import umap

from spotify_client import MySpotifyClient
import random
import time
import os
from dotenv import load_dotenv
import lyricsgenius as lg
import matplotlib.pyplot as plt

genius_client = os.getenv("GENIUS_CLINET")
genius_secret = os.getenv("GENIUS_SECRET")
genius_access = os.getenv("GENIUS_ACCESS_TOKEN")

def build_song_dataset(spotify_client, limit=100):
    seen = set()
    songs = []

    genius = lg.Genius(genius_access)

    def add_tracks(track_ids):
        for track_id in track_ids:
            if track_id not in seen:
                try: 
                    seen.add(track_id)
                    track = spotify_client.sp.track(track_id)
                    # audio_feat = spotify_client.sp.audio_features(track_id)[0] # audio_features has been depreciated rats
                    # if not audio_feat:
                    #     continue

                    # get audio features somehow in da future
                    # maybe follow the shazam youtube video idea

                    songs.append({
                        'track_id': track_id,
                        'name': track['name'],
                        'artist': track['artists'][0]['name'],
                        'artist_id': track['artists'][0]['id'],
                        'lyrics': genius.search_song(title=track['name'], artist=track['artists'][0]['name']).lyrics,
                        # **{k: audio_feat[k] for k in ['danceability', 'energy', 'key', 'loudness', 'mode', #This is super cool notation that I just found out about
                        #                              'speechiness', 'acousticness', 'instrumentalness',
                        #                              'liveness', 'valence', 'tempo']}
                    })
                except Exception as e:
                    print(f"Exception: {e}")
    

    # Get user's top tracks to base off of
    top_tracks = spotify_client.get_user_top_tracks(limit=limit)
    add_tracks(top_tracks)

    return pd.DataFrame(songs)


def embed_lyrics(df, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
    model = SentenceTransformer(model_name)
    df = df[df['lyrics'].notnull()]
    embeddings = model.encode(df['lyrics'].tolist(), show_progress_bar=True)
    df['lyrics_embedding'] = list(embeddings)
    return df


# def build_feature_matrix(df):
#     audio_cols = ['danceability', 'energy', 'key', 'loudness', 'mode',
#                   'speechiness', 'acousticness', 'instrumentalness',
#                   'liveness', 'valence', 'tempo']
    
#     audio_features = df[audio_cols].values
#     lyrics_embeddings = np.vstack(df['lyrics_embedding'].values)
    
#     # Normalize each part
#     scaler_audio = StandardScaler()
#     audio_scaled = scaler_audio.fit_transform(audio_features)

#     # Combine
#     combined = np.hstack([audio_scaled, lyrics_embeddings])
#     return combined


def cluster_songs(features, n_clusters=20):
    reducer = PCA(n_components=20)
    reduced = reducer.fit_transform(features)

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(reduced)

    return clusters, reduced


import matplotlib.pyplot as plt
from adjustText import adjust_text

def visualize(reduced_features, labels, df):
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.3, metric='euclidean')
    embedding = reducer.fit_transform(reduced_features)

    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab20', s=50, alpha=0.7)

    # Add labels for a few songs per cluster
    texts = []
    for i, (x, y) in enumerate(embedding):
        label = df.iloc[i]['name']
        cluster = labels[i]
        # Show only a few points per cluster to reduce clutter
        if random.random() < 0.1:  # 10% sample
            texts.append(plt.text(x, y, label[:20], fontsize=8, alpha=0.8))

    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    # Legend for clusters
    handles = [plt.Line2D([], [], marker='o', color='w',
                          label=f'Cluster {i}', markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=10)
               for i in range(len(set(labels)))]
    plt.legend(handles=handles, title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title('Song Clusters via UMAP')
    plt.xlabel('UMAP Dim 1')
    plt.ylabel('UMAP Dim 2')
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    load_dotenv()

    CLIENT_ID = os.getenv("MY_CLIENT_ID")
    CLIENT_SECRET = os.getenv("MY_CLIENT_SECRET")
    REDIRECT_URI = 'http://127.0.0.1:8000/callback'

    spotify_client = MySpotifyClient(CLIENT_ID, CLIENT_SECRET, REDIRECT_URI)

    print("Fetching songs...")
    df = build_song_dataset(spotify_client, limit=40)  # Start small

    print("Embedding lyrics...")
    df = embed_lyrics(df)

    features = np.vstack(df['lyrics_embedding'].values)

    # print("Building feature matrix...")
    # features = build_feature_matrix(df)

    print("Clustering...")
    labels, reduced = cluster_songs(features, n_clusters=10)

    df['cluster'] = labels
    print(df[['name', 'artist', 'cluster']].head(10))

    print("Visualizing...")
    # visualize(reduced, labels, df) #Have to improve visualization

