# Facial Emotion Detector 😃

A deep learning-based Facial Emotion Recognition system built using PyTorch and EfficientNet-B0, trained on the FER2013 dataset.

## Features

- Real-time emotion detection using webcam
- EfficientNet-B0 transfer learning
- Handles class imbalance using weighted loss
- Mixed precision training (AMP)
- Early stopping
- Automatic confusion matrix generation
- Training curve visualization

## Emotions Detected

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

---

## Dataset

This project uses the FER2013 dataset.

Dataset shape:

- Training Images: 28,709
- Test Images: 7,178
- Resolution: 48×48 grayscale

---

## Model Architecture

- Backbone: EfficientNet-B0
- Input Size: 224×224
- Framework: PyTorch
- Optimizer: AdamW
- Loss Function: Weighted Cross Entropy
- Mixed Precision Training (AMP)

---

## Results

### Test Accuracy

| Metric | Value |
|----------|----------|
| Accuracy | 69.8% |
| Macro F1 | 69% |

### Per-Class Performance

| Emotion | F1 Score |
|----------|----------|
| Angry | 0.61 |
| Disgust | 0.75 |
| Fear | 0.55 |
| Happy | 0.88 |
| Neutral | 0.66 |
| Sad | 0.57 |
| Surprise | 0.82 |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/facial-emotion-detector.git

cd facial-emotion-detector
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

Place the FER2013 dataset in the project directory.

Run:

```bash
python train.py
```

The script will:

- Train EfficientNet-B0
- Save best model weights
- Generate confusion matrix
- Generate training curves

---

## Webcam Inference

After training:

```bash
python webcam_test.py
```

Press:

```text
Q
```

to exit.

---

## Project Structure

```text
Facial-Emotion-Detector/
│
├── train.py
├── webcam_test.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   └── best_emotion_model.pth
│
├── outputs/
│   ├── confusion_matrix.png
│   └── training_curve.png
│
└── dataset/
    └── fer2013.csv
```

---

## Technologies Used

- Python
- PyTorch
- EfficientNet-B0
- OpenCV
- Scikit-Learn
- NumPy
- Matplotlib

---

## Future Improvements

- EfficientNet-B2/B3
- Focal Loss
- Test Time Augmentation (TTA)
- Vision Transformers (ViT)
- ONNX Export
- Streamlit Deployment

---

## License

MIT License

---

## Author

Your Name