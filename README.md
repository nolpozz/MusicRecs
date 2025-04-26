# MusicRecs
Algorithm for Cross-linguistic artist/song recommendation


Plan:
Scrap most of what's here

Step 1. Crappy frontend
Basic HTML page with a "recommend button"
enter the name of a song(that's been pulled) or an example lyric or describe the song
enter button

Step 2. Encoding Text
get spotify devs api
pull example ~500 songs
lyrics API to pull lyrics
encode the lyrics using a multilingual model
    song title or first couple of lines for now
store/cluster in a vector space

Step 3. Matching (content_based filtering_)
When a query is posted, encode the sentence or pull the song's encodings
match whichever 5 songs are closest in the vector space

Later:
expand to more songs
Add a collaborative filtering element
full lyric embeddings
    How to preserve the vector space/quality of embeddings with such long strings
    piece by piece and average?
    some secondary similarity scoring within a song then across?

later later:
encode the soundwaves of the songs for more than just textual recommendation
    look at sesame
    that one youtube video where the guy redoes pandora




