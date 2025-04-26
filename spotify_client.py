import spotipy
from spotipy.oauth2 import SpotifyOAuth

from flask import Flask, request
import threading
import webbrowser

class SpotifyClient:
    def __init__(self, client_id, client_secret, redirect_uri):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.sp = None
        self.auth_code = None

        self.authenticate()

    def authenticate(self):
        self.oauth = SpotifyOAuth(
            client_id=self.client_id,
            client_secret=self.client_secret,
            redirect_uri=self.redirect_uri,
            scope="user-library-read user-top-read",
            show_dialog=True,
            open_browser=False,  # Open browser manually, not automatically.
        )

        auth_url = self.oauth.get_authorize_url()

        print(f"Opening {auth_url} in your browser...")
        webbrowser.open(auth_url)

        app = Flask(__name__)

        @app.route('/callback')
        def callback():
            self.auth_code = request.args.get('code')
            threading.Thread(target=lambda: app.shutdown()).start()  # Shut down Flask server after receiving code
            return "Authorization successful! You can close this tab."

        def run_server():
            app.run(host='127.0.0.1', port=8000, debug=True, use_reloader = False)

        server_thread = threading.Thread(target=run_server)
        server_thread.start()

        while self.auth_code is None:
            pass

        token_info = self.oauth.get_access_token(self.auth_code)
        self.sp = spotipy.Spotify(auth=token_info['access_token'])

        print("Successfully authenticated with Spotify!")

    def get_user_top_artists(self, limit=20, time_range='medium_term'):
        results = self.sp.current_user_top_artists(limit=limit, time_range=time_range)
        return [artist['id'] for artist in results['items']]

    def get_artist_genres(self, artist_id):
        artist = self.sp.artist(artist_id)
        return artist['genres']

    def get_related_artists(self, artist_id):
        results = self.sp.artist_related_artists(artist_id)
        return [artist['id'] for artist in results['artists']]

    def get_user_top_tracks(self, limit=50, time_range='medium_term'):
        results = self.sp.current_user_top_tracks(limit=limit, time_range=time_range)
        return [track['id'] for track in results['items']]
    
