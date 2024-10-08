import streamlit as st
import cv2
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

from keras.models import load_model
model = load_model("/content/facial_expression_model.keras", compile=False)


# Spotify API credentials
SPOTIPY_CLIENT_ID = '84d4bb7deb364411a93cd1b55d3271b3'
SPOTIPY_CLIENT_SECRET = '10ead3f068e84213ac8d94a217abbcc9'
SPOTIPY_REDIRECT_URI = 'https://developer.spotify.com/dashboard/84d4bb7deb364411a93cd1b55d3271b3'

# Authenticate with Spotify
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID,
                                            client_secret=SPOTIPY_CLIENT_SECRET,
                                            redirect_uri=SPOTIPY_REDIRECT_URI,
                                            scope="user-library-read playlist-read-private"))

# Emotion labels based on the model's output
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Mapping emotions to Spotify music genres
emotion_to_genre = {
    'Angry': 'rock',
    'Disgust': 'alternative',
    'Fear': 'chill',
    'Happy': 'pop',
    'Sad': 'acoustic',
    'Surprise': 'electronic',
    'Neutral': 'classical'
}

# Function to predict emotion from the uploaded image
def predict_emotion(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (48, 48))

    # Preprocess the image for model prediction
    processed_image = resized_image.astype('float32') / 255.0
    processed_image = img_to_array(processed_image)
    processed_image = np.expand_dims(processed_image, axis=0)

    # Predict emotion
    prediction = model.predict(processed_image)
    emotion_idx = np.argmax(prediction)
    return emotion_labels[emotion_idx]

# Function to get Spotify song recommendations based on emotion
def recommend_songs(emotion):
    genre = emotion_to_genre.get(emotion, 'pop')  # Default to 'pop'

    # Search for playlists based on genre
    result = sp.search(q=f'genre:{genre}', type='playlist', limit=1)

    if result['playlists']['items']:
        playlist_id = result['playlists']['items'][0]['id']

        # Get tracks from the playlist
        tracks = sp.playlist_tracks(playlist_id)
        song_recommendations = []

        for item in tracks['items']:
            track = item['track']
            song_recommendations.append({
                'song_name': track['name'],
                'artist': track['artists'][0]['name'],
                'spotify_url': track['external_urls']['spotify']
            })

        return song_recommendations
    else:
        return []

# Streamlit Web App Interface
st.title("Emotion-Based Spotify Song Recommendation")
st.write("Capture an image using your webcam, and we'll recommend songs based on your emotion!")

# Capture image from webcam
webcam_image = st.camera_input("Take a picture")

if webcam_image is not None:
    # Convert the webcam image to an OpenCV image
    image = Image.open(webcam_image)
    image = np.array(image)

    # Display the captured image
    st.image(image, caption='Captured Image.', use_column_width=True)
    st.write("Classifying...")

    # Predict emotion from the captured image
    predicted_emotion = predict_emotion(image)
    st.write(f"Detected Emotion: {predicted_emotion}")

    # Recommend songs based on the detected emotion
    songs = recommend_songs(predicted_emotion)

    # Display song recommendations
    if songs:
        st.write(f"Recommended songs for {predicted_emotion} mood:")
        for song in songs:
            st.write(f"**{song['song_name']}** by {song['artist']}")
            st.markdown(f"[Listen on Spotify]({song['spotify_url']})", unsafe_allow_html=True)
    else:
        st.write("No songs found for this emotion.")

# !streamlit run /usr/local/lib/python3.10/dist-packages/colab_kernel_launcher.py

