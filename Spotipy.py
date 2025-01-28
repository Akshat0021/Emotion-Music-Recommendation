import spotipy
import spotipy.oauth2 as oauth2
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import time

# Authentication setup
# auth_manager = SpotifyClientCredentials(client_id='', client_secret='')
# sp = spotipy.Spotify(auth_manager=auth_manager)

# Function to get track IDs from a playlist
# def getTrackIDs(playlist_id):
#     track_ids = []
#     playlist = sp.playlist_tracks(playlist_id)
#     for item in playlist['items']:
#         track = item['track']
#         track_ids.append(track['id'])
#     return track_ids

# # Function to get track features
# def getTrackFeatures(track_id):
#     track_info = sp.track(track_id)
#     id= track_info['id'],
#     name = track_info['name']
#     album = track_info['album']['name']
#     artist = track_info['album']['artists'][0]['name']
#     cover = track_info['album']['images'][0]['url']
#     track_data = [id, name, album, artist, cover]
#     return track_data

# # Emotion dictionary and music distribution
# emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
# music_dist = {
#     0: "7r7GUDcrGuxmXvI7NAec9B", 1: "3buo6Vrf1tPWqyoq6JRAst", 2: "0leXHhrVXMT6ErPTuZ9Zy5", 
#     3: "1XboCzkhFLrKt5oyFaywMz", 4: "7yrysdedXxrV9fN8ZyBITB", 5: "2IJcAcRBAr9WidIAJhrFJM",
#     6: "2ktGZBEdJSWjZxlhBSWQ7J"
# }

# # Function to process and save tracks as CSV
# def process_and_save_tracks(emotion, playlist_id):
#     track_ids = getTrackIDs(playlist_id)
#     track_list = []
#     for track_id in track_ids:
#         time.sleep(0.3)  # Sleep to avoid hitting API rate limits
#         track_data = getTrackFeatures(track_id)
#         track_list.append(track_data)
    
#     # Convert to DataFrame and save to CSV
#     df = pd.DataFrame(track_list, columns=['Id', 'Name', 'Album', 'Artist', 'Cover'])
#     df.to_csv(f'songs/{emotion}.csv', index=False)  # Save to a separate CSV file
#     print(f"{emotion} CSV Generated")

# # Loop through all emotions and process each playlist
# for emotion, playlist_id in music_dist.items():
#     process_and_save_tracks(emotion_dict[emotion], playlist_id)