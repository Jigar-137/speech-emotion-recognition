🎧 Speech Emotion Recognition from Audio

This project detects human emotions from speech audio (.wav files) using machine learning and deep learning.
It supports both single emotion detection and emotion timeline analysis for long speech.

🔍 What This Project Can Do
🎯 Single Emotion Mode

Predicts the dominant emotion of the entire speech audio.

⏱ Multiple Emotions (Timeline Mode)

Splits long audio into 3-second chunks
Predicts emotion for each chunk
Shows how emotions change over time

😊 Emotions Detected

Happy
Sad
Angry
Calm
Fearful
Disgust
Neutral
Surprised

🧠 How It Works (Simple Flow)

User uploads a .wav audio file
MFCC features are extracted from speech
A trained LSTM model predicts emotion
Results are shown in a clean web interface

For long audio:
Audio is chunked → emotion predicted per segment → timeline shown

🛠️ Technologies Used

Python
Librosa (audio processing)
TensorFlow (LSTM model)
Scikit-learn
Flask (backend API)
HTML, CSS, JavaScript (frontend UI)

🖥️ How to Run the Project
pip install -r requirements.txt
python app.py


Open in browser:

http://127.0.0.1:5000


Upload a .wav file and select:
Single Emotion
Multiple Emotions (Timeline)

⚠️ Important Notes

The model predicts emotion based on acoustic features, not language meaning
Emotion labels reflect sound intensity and pitch
Timeline mode may show repeated emotions for intense speech
This behavior is expected and correct

📌 Project Purpose

This project was developed as part of an internship / academic project to learn:
Speech signal processing
Feature extraction (MFCC)
Deep learning with LSTM
Real-time ML deployment using Flask
UI + ML system integration

🚀 Future Improvements

Emotion smoothing in timeline
Dominant emotion summary
Audio playback synced with timeline
Improved real-world dataset support

👤 Author

Jigar
B.Tech (ICT) Student
