import spotipy
from spotipy.oauth2 import SpotifyOAuth

class SpotifyClient:
    def __init__(self, client_id, client_secret, redirect_uri):
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope="user-library-read user-top-read"
        ))

    def get_user_top_artists(self, limit=20, time_range='medium_term'):
        # Fetch user's top artists
        results = self.sp.current_user_top_artists(limit=limit, time_range=time_range)
        return [artist['id'] for artist in results['items']]

    def get_artist_genres(self, artist_id):
        # Fetch artist's genres
        artist = self.sp.artist(artist_id)
        return artist['genres']

    def get_related_artists(self, artist_id):
        # Fetch related artists
        results = self.sp.artist_related_artists(artist_id)
        return [artist['id'] for artist in results['artists']]

    def get_user_top_tracks(self, limit=50, time_range='medium_term'):
        # Fetch user's top tracks
        results = self.sp.current_user_top_tracks(limit=limit, time_range=time_range)
        return [track['id'] for track in results['items']]
