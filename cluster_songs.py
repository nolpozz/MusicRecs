import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances
import random

from spotify_client import MySpotifyClient
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

                    lyrics = genius.search_song(title=track['name'], artist=track['artists'][0]['name']).lyrics
                    lyrics = lyrics.replace("Contributors", "").replace("Lyrics", "")

                    songs.append({
                        'track_id': track_id,
                        'name': track['name'],
                        'artist': track['artists'][0]['name'],
                        'artist_id': track['artists'][0]['id'],
                        'lyrics': lyrics,
                        'popularity': track['popularity'],
                        'genres': list(set(
                            genre
                            for artist in track['artists']
                            for genre in spotify_client.get_artist_genres(artist['id'])
                        )) # approximation
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

def expand_features(df):
    # One-hot encode genres
    all_genres = sorted(set(g for genre_list in df['genres'] for g in genre_list))
    for genre in all_genres:
        df[f"genre_{genre}"] = df["genres"].apply(lambda g_list: int(genre in g_list))

    # Stack all features
    lyrics_features = np.vstack(df['lyrics_embedding'])
    popularity_features = df[['popularity']].values
    genre_features = df[[f"genre_{g}" for g in all_genres]].values

    combined = np.hstack([lyrics_features, popularity_features, genre_features])
    return combined


def cluster_songs(features, n_clusters=10):
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
        representative_song = df.iloc[closest_index]

        representatives.append({
            'cluster': cluster_id,
            'name': representative_song['name'],
            'artist': representative_song['artist'],
            'lyrics': representative_song['lyrics'][:500] + '...',
        })

    return representatives


def summarize_clusters(df, n_keywords=10):
    summaries = []
    for cluster_id in sorted(df['cluster'].unique()):
        # cluster_lyrics = df[df['cluster'] == cluster_id]['lyrics']
        vectorizer = TfidfVectorizer(stop_words='english', max_features=n_keywords)
        # tfidf_matrix = vectorizer.fit_transform(cluster_lyrics)
        top_keywords = vectorizer.get_feature_names_out()
        summaries.append((cluster_id, top_keywords.tolist()))
    return summaries


def cluster_songs(features, n_clusters=20):
    reducer = PCA(n_components=20)
    reduced = reducer.fit_transform(features)

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(reduced)

    return clusters, reduced


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from adjustText import adjust_text

def visualize(reduced_features, labels, df):
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
    embedding = tsne.fit_transform(reduced_features)

    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab20', s=50, alpha=0.7)

    # Add text labels (sampled)
    texts = []
    for i, (x, y) in enumerate(embedding):
        if random.random() < 0.1:  # 10% of points
            label = df.iloc[i]['name']
            texts.append(plt.text(x, y, label[:20], fontsize=8, alpha=0.8))

    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    # Legend for clusters
    handles = [plt.Line2D([], [], marker='o', color='w',
                          label=f'Cluster {i}', markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=10)
               for i in range(len(set(labels)))]
    plt.legend(handles=handles, title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title('Song Clusters via t-SNE')
    plt.xlabel('t-SNE Dim 1')
    plt.ylabel('t-SNE Dim 2')
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
    print(df["lyrics"][0])

    # features = np.vstack(df['lyrics_embedding'].values)

    # print("Building feature matrix...")
    # features = build_feature_matrix(df)

    print("Expanding features...")
    features = expand_features(df)

    print("Clustering...")
    labels, reduced = cluster_songs(features, n_clusters=10)
    df['cluster'] = labels

    print("Summarizing clusters...")
    cluster_themes = summarize_clusters(df)
    for cid, keywords in cluster_themes:
        print(f"Cluster {cid}: {', '.join(keywords)}")

    print("Getting representatives...")
    reps = get_representative_songs(reduced, labels, df)
    for rep in reps:
        print(f"Cluster {rep['cluster']} Representative: {rep['name']} by {rep['artist']}")

    print("Visualizing...")
    visualize(reduced, labels, df) #Have to improve visualization

    print("Saving Clusters...")
    df.to_csv("clustered_songs.csv")