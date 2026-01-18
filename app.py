import os
import logging
import traceback

from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

# Feature extraction utilities
from utils.feature_extraction import (
    prepare_input,
    extract_mfcc_chunks
)

# --------------------------------------------------
# BASIC CONFIGURATION
# --------------------------------------------------

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# --------------------------------------------------
# LOAD MODEL & LABEL ENCODER
# --------------------------------------------------

MODEL_PATH = "models/emotion_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

try:
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    logging.info("Model and Label Encoder loaded successfully")
except Exception as e:
    logging.error("Error loading model or label encoder")
    logging.error(traceback.format_exc())
    raise e

# --------------------------------------------------
# ROUTES
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    """
    Load main UI page
    """
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint
    """
    return jsonify({
        "status": "OK",
        "message": "Emotion Recognition API running"
    })


# --------------------------------------------------
# SINGLE EMOTION PREDICTION
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict_emotion():
    """
    Predict ONE dominant emotion from entire audio
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "Empty file name"}), 400

        if not file.filename.lower().endswith(".wav"):
            return jsonify({"error": "Only .wav files are supported"}), 400

        # Save temporary file
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, file.filename)
        file.save(file_path)

        # Feature extraction
        mfcc = prepare_input(file_path)

        # Prediction
        prediction = model.predict(mfcc)
        emotion_index = int(np.argmax(prediction))
        emotion = label_encoder.inverse_transform([emotion_index])[0]

        # Cleanup
        os.remove(file_path)

        return jsonify({
            "mode": "single",
            "emotion": emotion
        })

    except Exception as e:
        logging.error("Single emotion prediction failed")
        logging.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# --------------------------------------------------
# MULTI EMOTION (AUDIO CHUNKING)
# --------------------------------------------------

@app.route("/predict_chunks", methods=["POST"])
def predict_emotion_chunks():
    """
    Predict MULTIPLE emotions from long audio
    using audio chunking
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "Empty file name"}), 400

        if not file.filename.lower().endswith(".wav"):
            return jsonify({"error": "Only .wav files are supported"}), 400

        # Save temporary file
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, file.filename)
        file.save(file_path)

        # Extract MFCC chunks
        mfcc_chunks = extract_mfcc_chunks(file_path)

        results = []
        chunk_duration = 3  # seconds
        current_time = 0

        for mfcc in mfcc_chunks:
            mfcc = mfcc.reshape(1, 1, mfcc.shape[0])
            prediction = model.predict(mfcc)
            emotion_index = int(np.argmax(prediction))
            emotion = label_encoder.inverse_transform(
                [emotion_index]
            )[0]

            results.append({
                "start_time": f"{current_time}s",
                "end_time": f"{current_time + chunk_duration}s",
                "emotion": emotion
            })

            current_time += chunk_duration

        # Cleanup
        os.remove(file_path)

        return jsonify({
            "mode": "multi",
            "chunk_duration": f"{chunk_duration} seconds",
            "total_chunks": len(results),
            "results": results
        })

    except Exception as e:
        logging.error("Multi emotion prediction failed")
        logging.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# --------------------------------------------------
# MAIN ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
