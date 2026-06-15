# Deepfake Audio Detection

## Overview

This project presents a machine learning based system for detecting AI-generated speech and distinguishing it from genuine human speech recordings. The system utilizes advanced audio signal processing techniques and feature-based machine learning to classify audio samples as Genuine or Deepfake.

## Dataset

The model was developed using the Fake-or-Real (FoR) Dataset, specifically the for-norm version. The dataset contains both genuine human speech recordings and AI-generated speech samples that are balanced and normalized for consistent training and evaluation.

## Audio Preprocessing

Audio files were resampled to 16 kHz.

Each recording was standardized to a duration of 2 seconds.

Short recordings were padded and longer recordings were truncated to ensure uniform input length.

## Feature Extraction

MFCC

Delta MFCC

Chroma Features

Spectral Contrast

Tonnetz Features

Spectral Centroid

Spectral Bandwidth

Spectral Rolloff

Zero Crossing Rate

RMS Energy

These features capture spectral, temporal and perceptual characteristics of speech and help identify artifacts commonly present in synthetic audio.

## Classification Model

A Random Forest classifier was trained using the extracted feature vectors after feature normalization using StandardScaler.

Model Parameters:

Number of Trees: 600

Maximum Depth: 30

Minimum Samples Split: 4

Minimum Samples Leaf: 2

Random State: 42

## Evaluation Metrics

Accuracy

F1 Score

Equal Error Rate (EER)

Confusion Matrix

Precision

Recall

## Results

Accuracy: 97.47%

F1 Score: 97.50%

Equal Error Rate (EER): 2.93%

Confusion Matrix:

[721 29]

[9 741]

Genuine Speech Precision: 0.99

Genuine Speech Recall: 0.96

Genuine Speech F1 Score: 0.97

Deepfake Speech Precision: 0.96

Deepfake Speech Recall: 0.99

Deepfake Speech F1 Score: 0.97

## Repository Contents

app.py

predict.py

requirements.txt

scaler.pkl

deepfake_audio_detection.ipynb

README.md

## Running the Project

Install the required dependencies using requirements.txt.

Run the Streamlit application using:

streamlit run app.py

Upload a WAV audio file through the interface.

The application returns the predicted class and confidence score.

## Applications

Deepfake Audio Detection

Voice Authentication Systems

Digital Media Verification

Audio Forensics

Cybersecurity Applications

Fraud Detection

## Future Work

Integration with the ASVspoof 2019 dataset.

Cross-dataset evaluation and benchmarking.

Ensemble learning approaches.

Real-time inference optimization.

Transformer-based audio classification models.

## Author

Shreya Kale

Indian Institute of Technology Roorkee

