import streamlit as st
import numpy as np
import librosa
import soundfile
from sklearn.ensemble import RandomForestClassifier
import os
import pickle

# Dummy classifier training (you can replace this with a trained model)
# This is only for demo purposes

def train_dummy_classifier():
    # Let's simulate a trained model
    X_train = np.random.rand(100, 40)  # 40 MFCC features
    y_train = np.random.choice(['happy', 'sad', 'angry', 'neutral'], 100)
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Save model to file
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    return model

# Load or train model
if os.path.exists("model.pkl"):
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
else:
    model = train_dummy_classifier()


# Feature extraction function
def extract_features(file_path):
    with soundfile.SoundFile(file_path) as sound_file:
        audio_data = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
    return mfccs_mean

# Streamlit UI

st.title("🎙️ Speech Emotion Recognition")
st.write("Upload an audio file (wav format) to predict emotion")

uploaded_file = st.file_uploader("Upload Audio File", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    features = extract_features("temp.wav")
    features = features.reshape(1, -1)
    
    prediction = model.predict(features)
    
    st.success(f"Predicted Emotion: **{prediction[0]}**")
