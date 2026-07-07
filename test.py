import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# ==========================
# CONFIG
# ==========================

MODEL_PATH = "checkpoints/best_model.pth"

EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using:", DEVICE)

# ==========================
# MODEL
# ==========================

model = models.efficientnet_b0(weights=None)

in_features = model.classifier[1].in_features

model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features, 7)
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.to(DEVICE)
model.eval()

print("Model Loaded Successfully")

# ==========================
# FACE DETECTOR
# ==========================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# ==========================
# TRANSFORMS
# ==========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================
# WEBCAM
# ==========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

print("\nPress Q to quit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(40, 40)
    )

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]

        pil_img = Image.fromarray(face)

        img_tensor = transform(
            pil_img
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(img_tensor)

            probs = torch.softmax(
                output,
                dim=1
            )

            confidence, pred = torch.max(
                probs,
                dim=1
            )

        emotion = EMOTIONS[pred.item()]
        conf = confidence.item() * 100

        label = f"{emotion} ({conf:.1f}%)"

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            label,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "Emotion Detection",
        frame
    )

    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()