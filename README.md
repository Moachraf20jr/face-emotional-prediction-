# Facial Emotion Detection

Real-time facial emotion detection using OpenCV and TensorFlow/Keras. Detects 7 emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
python train_model.py
```

## Run Real-Time Detection

```bash
python real_time_detection.py
```

Press `q` to quit.

## Project Structure

```
├── dataset/                          # Training data
├── models/
│   ├── emotion_model_best.h5         # Trained Keras model
│   └── training_history.png          # Training curves
├── real_time_detection.py            # Real-time webcam inference
├── train_model.py                    # Training script
├── requirements.txt
└── .gitignore
```

## Emotions Detected

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise
