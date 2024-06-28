from flask import Flask, request, jsonify, render_template
import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
import random

app = Flask(__name__)

# Load the trained model
model = load_model("my_model.h5")

# Function definitions from efficient-net.py
# ...

@app.route('/')
def index():
    return render_template('detect.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        video_path = os.path.join('uploads', file.filename)
        file.save(video_path)
        prediction = predict_video(video_path)
        return jsonify({'prediction': prediction})
    
def crop_faces(frame):
    # Load the Haar cascade for face detectionp
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Crop faces from the frame
    cropped_faces = []
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        cropped_faces.append(face)

    return cropped_faces

def extract_frames(video_path, num_frames=100):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Choose 70 random frame indices
    random_indices = random.sample(range(frame_count), num_frames)
    
    for idx in random_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Crop faces from the frame
        cropped_faces = crop_faces(frame)

        # Resize and save each cropped face
        for face in cropped_faces:
            if face.shape[0] > 0 and face.shape[1] > 0:
                resized_face = cv2.resize(face, (128, 128))
                frames.append(resized_face)

    cap.release()
    return frames

def predict_video(video_path):
    # Extract frames from the video
    frames = extract_frames(video_path)

    if len(frames) == 0:
        return "No faces detected in the video."

    # Convert frames to numpy array and preprocess for prediction
    frames = np.array(frames) / 255.0

    # Predict using the model
    predictions = model.predict(frames)

    # Take the average prediction across frames
    average_prediction = np.mean(predictions)

    # Output the prediction result
    if average_prediction <= 0.5:
        return "Deepfake (Confidence: {:.2f}%)".format((1-average_prediction) * 100)
    else:
        return "Real (Confidence: {:.2f}%)".format((average_prediction) * 100)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
