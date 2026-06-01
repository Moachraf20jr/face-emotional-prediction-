import cv2
import numpy as np
import os
from collections import deque

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

EMOTION_COLORS = {
    'Angry':    (0, 0, 255),
    'Disgust':  (0, 165, 255),
    'Fear':     (128, 0, 128),
    'Happy':    (0, 255, 0),
    'Neutral':  (200, 200, 200),
    'Sad':      (255, 0, 0),
    'Surprise': (255, 255, 0)
}

# ── Load model ────────────────────────────────────────────────────────────────
USE_H5 = os.path.exists('models/emotion_model_best.h5')

if USE_H5:
    import tensorflow as tf
    model = tf.keras.models.load_model('models/emotion_model_best.h5')
    print("Loaded H5 model (emotion_model_best.h5)")

    def predict_emotion(roi):
        roi = roi.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)
        return model.predict(roi, verbose=0)[0]

else:
    print("Error: No model found. Run train_model.py first.")
    exit()

# ── Load Haar Cascade ─────────────────────────────────────────────────────────
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_classifier = cv2.CascadeClassifier(cascade_path)
if face_classifier.empty():
    print("Error: Haar cascade not loaded.")
    exit()

# ── Temporal smoothing ────────────────────────────────────────────────────────
SMOOTHING_FRAMES = 5
prediction_history = deque(maxlen=SMOOTHING_FRAMES)

# ── Open webcam ───────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

frame_count = 0
fps_start = cv2.getTickCount()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if roi_gray.sum() != 0:
            prediction = predict_emotion(roi_gray)
            prediction_history.append(prediction)

            avg_prediction = np.mean(prediction_history, axis=0)
            label = EMOTION_LABELS[avg_prediction.argmax()]
            confidence = avg_prediction.max() * 100

            color = EMOTION_COLORS.get(label, (255, 255, 255))

            display_text = f"{label} ({confidence:.1f}%)"
            label_y = y - 10 if y - 10 > 25 else y + h + 25

            cv2.rectangle(frame, (x, label_y - 25), (x + 220, label_y + 5), color, -1)
            cv2.putText(frame, display_text, (x + 5, label_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            bar_width = int(confidence * 2)
            cv2.rectangle(frame, (x, y + h + 8), (x + 200, y + h + 18), (50, 50, 50), -1)
            cv2.rectangle(frame, (x, y + h + 8), (x + bar_width, y + h + 18), color, -1)

    # ── FPS counter ───────────────────────────────────────────────────────────
    frame_count += 1
    elapsed = (cv2.getTickCount() - fps_start) / cv2.getTickFrequency()
    if elapsed >= 1.0:
        fps = frame_count / elapsed
        frame_count = 0
        fps_start = cv2.getTickCount()
    else:
        fps = frame_count / max(elapsed, 0.001)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(frame, "Press 'q' to quit", (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow('Facial Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
