from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def recommend_content_based(artist_id, tfidf_matrix, artist_index, top_n=10):
    # Get the index of the artist in the DataFrame
    idx = artist_index[artist_id]

    # Compute the cosine similarity of the artist with all other artists
    cosine_similarities = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    # Get the top N similar artists
    similar_indices = cosine_similarities.argsort()[:-top_n-1:-1]

    return similar_indices
