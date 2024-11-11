import time
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import pandas as pd

df = pd.read_csv('data.csv')

# Select all numerical features for recommendation, including duration
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
numerical_features.remove('explicit')  # Assuming explicit is binary, we can exclude or include based on preference

# Standardize the features
scaler = StandardScaler()
song_data_scaled = scaler.fit_transform(df[numerical_features])

# Fit the Nearest Neighbors model
knn_model = NearestNeighbors(n_neighbors=10, metric='cosine')  # Using more neighbors to consider duration filtering
knn_model.fit(song_data_scaled)

def recommend_next_song(song_name, played_songs, duration_tolerance=10000):
    """
    Recommends the next song to play after the given song, based on similarity and duration closeness.

    Parameters:
    - song_name (str): The name of the song currently playing.
    - played_songs (set): Set of song names that have already been played.
    - duration_tolerance (int): The maximum allowed difference in duration (in milliseconds).

    Returns:
    - str, int: The recommended song name and its duration in milliseconds.
    """
    if song_name not in df['name'].values:
        print("Song not found in the dataset.")
        return None, None

    # Get the index and duration of the current song
    song_idx = df[df['name'] == song_name].index[0]
    current_duration = df.loc[song_idx, 'duration_ms']

    # Find the nearest neighbors
    distances, indices = knn_model.kneighbors([song_data_scaled[song_idx]], n_neighbors=10)

    # Filter songs by duration closeness and whether they have been played
    for idx in indices[0][1:]:  # Skip the first result as it is the current song itself
        recommended_song_name = df.loc[idx, 'name']
        recommended_song_duration = df.loc[idx, 'duration_ms']
        
        # Check if the song has not been played and is within duration tolerance
        if recommended_song_name not in played_songs and abs(recommended_song_duration - current_duration) <= duration_tolerance:
            return recommended_song_name, recommended_song_duration

    print("No suitable recommendation found within duration tolerance.")
    return None, None

# Continuous recommendation loop
current_song = "Danny Boy"  # Replace with any song name from the dataset
duration_tolerance = 10000  # Duration tolerance in milliseconds
played_songs = set()

while current_song:
    # If all songs have been played, reset the played songs set
    if len(played_songs) == len(df):
        print("All songs have been played. Resetting playlist.")
        played_songs.clear()
    
    # Recommend the next song and get its duration
    next_song, next_duration = recommend_next_song(current_song, played_songs, duration_tolerance=duration_tolerance)
    
    if not next_song or not next_duration:
        print("Ending recommendation loop - no more songs within the specified duration tolerance.")
        break
    
    print(f"Now playing: '{current_song}', Next song: '{next_song}'")

    # Mark the song as played and add to the set
    played_songs.add(current_song)

    # Sleep for the duration of the current song (convert milliseconds to seconds)
    time.sleep(next_duration / 20000)
    
    # Update the current song to the next one for the loop to continue
    current_song = next_song
