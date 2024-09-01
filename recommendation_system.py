import numpy as np

from spotify_client import SpotifyClient
from data_processing import preprocess_data, vectorize_genres
from content_based import recommend_content_based
from collaborative_filtering import train_als_model, recommend_collaborative

def hybrid_recommendation_system(user_id, spotify_client, top_n=10):
    # Preprocess data
    df = preprocess_data(spotify_client)
    tfidf_matrix = vectorize_genres(df)

    # Create artist index mapping
    artist_index = pd.Series(df.index, index=df['artist_id']).to_dict()

    # Content-based recommendations
    content_recommendations = recommend_content_based(user_id, tfidf_matrix, artist_index, top_n=top_n)

    # Collaborative filtering recommendations
    model, user_artist_matrix = train_als_model(df_user_artists)  # df_user_artists should be preprocessed
    collaborative_recommendations = recommend_collaborative(model, user_artist_matrix, user_id, top_n=top_n)

    # Combine the results (weighted sum approach)
    final_recommendations = np.concatenate([content_recommendations, collaborative_recommendations])
    unique_recommendations = np.unique(final_recommendations)

    return unique_recommendations[:top_n]

if __name__ == "__main__":
    # Example usage
    CLIENT_ID = 'your_spotify_client_id'
    CLIENT_SECRET = 'your_spotify_client_secret'
    REDIRECT_URI = 'http://localhost:8888/callback'

    spotify_client = SpotifyClient(CLIENT_ID, CLIENT_SECRET, REDIRECT_URI)
    
    user_id = 1  # Replace with the actual user ID
    recommendations = hybrid_recommendation_system(user_id, spotify_client)
    print("Recommended Artists:", recommendations)
