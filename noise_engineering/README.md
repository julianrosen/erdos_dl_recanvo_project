# Noise engineering

This folder contains some of the notebooks and utilities we used for experimentation with manipulating background noise.

- `add_noise.py` contains some utility functions used in some of the notebooks. Their role is to extract features from a given data (sub)set using any model, while adding background noise from a given set in "real time".
- `session_classifier.ipynb` is our experiment at classifying vocalizations by *session* instead of *label*.
- `channel_testing.ipynb` carries out some exploratory analysis by studying whether analyzing stereo channels separately, or combining them creatively, might lead to interesting results, particularly towards removing far-away noises.
- `noisy_classifier_f1.ipynb` compares performance of the Hubert + logistic regression model with and without noise addition.
- `denoised_cnn.ipynb` uses the denoiser model on 4 CNN layers and 2 FC layers for classification for P01
- `hubert_mel_hybrid.ipynb` combines both the mel spectogram features from the 4 CNN layers along with hubert into the 2 FC layers 
