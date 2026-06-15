import sys
import librosa
import numpy as np
import joblib

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


audio_file = sys.argv[1]

features = extract_features(audio_file)

features = scaler.transform([features])

prediction = model.predict(features)[0]

confidence = np.max(
    model.predict_proba(features)
)

label = (
    "Deepfake"
    if prediction == 1
    else "Genuine"
)

print(f"Prediction : {label}")
print(f"Confidence : {confidence:.4f}")
# paste complete predict.py code here
