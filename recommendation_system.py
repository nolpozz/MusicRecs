import numpy as np
import pandas as pd

from spotify_client import SpotifyClient
from data_processing import preprocess_data, vectorize_genres
from content_based import recommend_content_based
from collaborative_filtering import train_als_model, recommend_collaborative

import os
from dotenv import load_dotenv

def hybrid_recommendation_system(spotify_client, top_n=10):

    sp = spotify_client
    top_artists = sp.get_user_top_artists()


    df = preprocess_data(spotify_client)
    tfidf_matrix = vectorize_genres(df)

    # Create artist index mapping
    artist_index = pd.Series(df.index, index=df['artist_id']).to_dict()
    index_artist = {v: k for k, v in artist_index.items()}


    # Content-based recommendations
    print(sp.get_artist_name(top_artists[1]))
    content_recommendations = recommend_content_based(top_artists[1], tfidf_matrix, artist_index, top_n=top_n)

    # Collaborative filtering recommendations
    # model, user_artist_matrix = train_als_model(df_user_artists)  # df_user_artists should be preprocessed
    # collaborative_recommendations = recommend_collaborative(model, user_artist_matrix, user_id, top_n=top_n)

    # Combine the results (weighted sum approach)
    # final_recommendations = np.concatenate([content_recommendations, collaborative_recommendations])
    # unique_recommendations = np.unique(final_recommendations)

    # return unique_recommendations[:top_n]
    recs = []
    for i in range(top_n):
        recs.append(sp.get_artist_name(index_artist.get(content_recommendations[i])))
    return recs

if __name__ == "__main__":

    load_dotenv()

    CLIENT_ID = os.getenv("MY_CLIENT_ID")
    CLIENT_SECRET = os.getenv("MY_CLIENT_SECRET")
    REDIRECT_URI = 'http://127.0.0.1:8000/callback'

    spotify_client = SpotifyClient(CLIENT_ID, CLIENT_SECRET, REDIRECT_URI)

    recommendations = hybrid_recommendation_system(spotify_client)
    print("Recommended Artists:", recommendations)
