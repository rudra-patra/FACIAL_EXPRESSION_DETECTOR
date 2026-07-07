import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ==========================
# DATASET PATHS
# ==========================

TRAIN_DIR = "archive/train"
TEST_DIR = "archive/test"

# ==========================
# EMOTION LABELS
# ==========================

emotion_map = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprise": 6
}

# ==========================
# COUNT IMAGES
# ==========================

print("\n========== DATASET OVERVIEW ==========\n")

for emotion in os.listdir(TRAIN_DIR):
    path = os.path.join(TRAIN_DIR, emotion)

    if os.path.isdir(path):
        count = len(os.listdir(path))
        print(f"{emotion:<10} : {count} images")

# ==========================
# DISPLAY CLASS DISTRIBUTION
# ==========================

class_counts = {}

for emotion in os.listdir(TRAIN_DIR):
    path = os.path.join(TRAIN_DIR, emotion)

    if os.path.isdir(path):
        class_counts[emotion] = len(os.listdir(path))

plt.figure(figsize=(8, 5))
plt.bar(class_counts.keys(), class_counts.values())
plt.title("Training Dataset Distribution")
plt.xlabel("Emotion")
plt.ylabel("Number of Images")
plt.xticks(rotation=45)
plt.show()

# ==========================
# DISPLAY SAMPLE IMAGES
# ==========================

print("\nDisplaying sample images...\n")

plt.figure(figsize=(12, 6))

for idx, emotion in enumerate(emotion_map.keys()):
    folder = os.path.join(TRAIN_DIR, emotion)

    sample_image = os.listdir(folder)[0]

    img_path = os.path.join(folder, sample_image)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    plt.subplot(2, 4, idx + 1)
    plt.imshow(img, cmap="gray")
    plt.title(emotion)
    plt.axis("off")

plt.tight_layout()
plt.show()

# ==========================
# LOAD DATASET
# ==========================

X_train = []
y_train = []

print("\nLoading training images...\n")

for emotion in emotion_map.keys():

    folder = os.path.join(TRAIN_DIR, emotion)

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = cv2.resize(img, (48, 48))

        img = img / 255.0

        X_train.append(img)

        y_train.append(emotion_map[emotion])

X_train = np.array(X_train)
y_train = np.array(y_train)

# ==========================
# LOAD TEST DATA
# ==========================

X_test = []
y_test = []

print("Loading testing images...\n")

for emotion in emotion_map.keys():

    folder = os.path.join(TEST_DIR, emotion)

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = cv2.resize(img, (48, 48))

        img = img / 255.0

        X_test.append(img)

        y_test.append(emotion_map[emotion])

X_test = np.array(X_test)
y_test = np.array(y_test)

# ==========================
# CNN INPUT SHAPE
# ==========================

X_train = X_train.reshape(-1, 48, 48, 1)
X_test = X_test.reshape(-1, 48, 48, 1)

# ==========================
# IMPORTANT DATA INFORMATION
# ==========================

print("\n========== DATA SUMMARY ==========\n")

print("Training Images Shape :", X_train.shape)
print("Training Labels Shape :", y_train.shape)

print("Testing Images Shape  :", X_test.shape)
print("Testing Labels Shape  :", y_test.shape)

print("\nPixel Value Range")

print("Minimum Pixel :", X_train.min())
print("Maximum Pixel :", X_train.max())

print("\nUnique Classes")

for emotion, label in emotion_map.items():
    print(f"{label} -> {emotion}")

print("\nClass Distribution")

counter = Counter(y_train)

for label, count in sorted(counter.items()):
    emotion_name = list(emotion_map.keys())[label]
    print(f"{emotion_name:<10} : {count}")

# ==========================
# CHECK FOR MISSING VALUES
# ==========================

print("\nMissing Values Check")

print("NaN in X_train :", np.isnan(X_train).sum())
print("NaN in X_test  :", np.isnan(X_test).sum())

# ==========================
# SAVE NUMPY FILES
# ==========================

os.makedirs("data/processed", exist_ok=True)

np.save("data/processed/X_train.npy", X_train)
np.save("data/processed/y_train.npy", y_train)

np.save("data/processed/X_test.npy", X_test)
np.save("data/processed/y_test.npy", y_test)

print("\nProcessed files saved successfully.")
print("Ready for CNN training.")