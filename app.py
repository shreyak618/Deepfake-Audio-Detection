import streamlit as st
import librosa
import numpy as np
import joblib
import tempfile

model = joblib.load("model_compressed.pkl")
scaler = joblib.load("scaler.pkl")


def extract_features(file_path):

    y, sr = librosa.load(
        file_path,
        sr=16000,
        duration=2
    )

    target_len = 32000

    if len(y) < target_len:
        y = np.pad(
            y,
            (0, target_len - len(y))
        )
    else:
        y = y[:target_len]

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=20
    )

    delta = librosa.feature.delta(mfcc)

    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr
    )

    contrast = librosa.feature.spectral_contrast(
        y=y,
        sr=sr
    )

    tonnetz = librosa.feature.tonnetz(
        y=librosa.effects.harmonic(y),
        sr=sr
    )

    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )

    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr
    )

    zcr = librosa.feature.zero_crossing_rate(y)

    rms = librosa.feature.rms(y=y)

    features = []

    for feat in [
        mfcc,
        delta,
        chroma,
        contrast,
        tonnetz,
        centroid,
        bandwidth,
        rolloff,
        zcr,
        rms
    ]:

        features.append(feat.mean())
        features.append(feat.std())

        if feat.shape[0] > 1:
            features.extend(feat.mean(axis=1))
            features.extend(feat.std(axis=1))

    return np.array(features)


st.title("Deepfake Audio Detection")

uploaded_file = st.file_uploader(
    "Upload WAV File",
    type=["wav"]
)

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as tmp:

        tmp.write(uploaded_file.read())
        path = tmp.name

    feat = extract_features(path)

    feat = scaler.transform([feat])

    pred = model.predict(feat)[0]

    prob = np.max(
        model.predict_proba(feat)
    )

    label = (
        "Deepfake"
        if pred == 1
        else "Genuine"
    )

    st.success(f"Prediction: {label}")
    st.write(f"Confidence: {prob:.4f}")
# paste complete app.py code here
