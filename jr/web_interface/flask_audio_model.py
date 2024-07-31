from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
import torchaudio
import pickle
import pandas as pd

app = Flask(__name__)

# I was getting an error "Cross-Origin Request Blocked" in my
# browser, this code is to fix.
cors = CORS(app, resources={"/*": {"origins": "*"}})
app.config["CORS_HEADERS"] = "content-type"


@app.route("/", methods=["GET", "POST", "DELETE"])
def add_numbers():
    waveform, sample_rate = torchaudio.load(request.data)
    bundle = torchaudio.pipelines.HUBERT_BASE
    model = bundle.get_model()
    waveform = torchaudio.functional.resample(
        waveform, sample_rate, bundle.sample_rate
    )
    with torch.no_grad():
        features, _ = model.extract_features(waveform)
    features = features[0].mean((0, 1))
    with open("../model.pkl", "rb") as f:
        my_model = pickle.load(f)
        labels = pickle.load(f)
    predictions = pd.Series(
        my_model.predict_proba(features.unsqueeze(0))[0], labels
    )
    return jsonify(
        result=predictions.sort_values(ascending=False)
        .round(3)
        .to_frame()
        .to_html(header=False)
    )
