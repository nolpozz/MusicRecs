import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_data(spotify_client):
    # Get user top artists
    user_top_artists = spotify_client.get_user_top_artists()

    # Get genres and create a DataFrame
    artist_genres = [(artist_id, spotify_client.get_artist_genres(artist_id)) for artist_id in user_top_artists]
    df = pd.DataFrame(artist_genres, columns=['artist_id', 'genres'])

    # Create genre string for each artist
    df['genre_str'] = df['genres'].apply(lambda genres: ' '.join(genres))

    return df

def vectorize_genres(df):
    # Vectorize the genres using TF-IDF
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['genre_str'])
    return tfidf_matrix
