import numpy as np
from implicit.als import AlternatingLeastSquares
import scipy.sparse as sparse

def train_als_model(df_user_artists):
    # Create a sparse matrix for user-item interactions
    user_artist_matrix = sparse.csr_matrix((df_user_artists['play_count'], 
                                            (df_user_artists['user_id'], df_user_artists['artist_id'])))

    # Initialize ALS model
    model = AlternatingLeastSquares(factors=50, regularization=0.1, iterations=30)

    # Fit the model
    model.fit(user_artist_matrix)

    return model, user_artist_matrix

def recommend_collaborative(model, user_artist_matrix, user_id, top_n=10):
    # Recommend artists for a given user using the ALS model
    user_items = user_artist_matrix.T.tocsr()
    recommendations = model.recommend(user_id, user_items, N=top_n, filter_already_liked_items=True)
    
    return recommendations
