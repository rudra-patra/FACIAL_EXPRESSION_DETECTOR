
import os
import numpy as np
from collections import Counter

import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

from torchvision import transforms
from torchvision.models import efficientnet_b0

from PIL import Image

from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt

# =====================================================
# GPU SETTINGS
# =====================================================

torch.backends.cudnn.benchmark = True

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nUsing Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            2
        ),
        "GB"
    )

# =====================================================
# DATASET
# =====================================================

class FERDataset(Dataset):

    def __init__(
        self,
        images,
        labels,
        transform=None
    ):

        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = self.images[idx]

        image = (
            image.squeeze() * 255
        ).astype(np.uint8)

        image = Image.fromarray(image)

        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label


# =====================================================
# LOAD DATA
# =====================================================

print("\nLoading data...")

X_train = np.load("data/processed/X_train.npy")
y_train = np.load("data/processed/y_train.npy")

X_test = np.load("data/processed/X_test.npy")
y_test = np.load("data/processed/y_test.npy")

print("Train:", X_train.shape)
print("Test :", X_test.shape)

# =====================================================
# CLASS NAMES
# =====================================================

class_names = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

# =====================================================
# AUGMENTATION
# =====================================================

train_transform = transforms.Compose([
    transforms.Resize((224,224)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# =====================================================
# DATASET
# =====================================================

train_dataset = FERDataset(
    X_train,
    y_train,
    train_transform
)

test_dataset = FERDataset(
    X_test,
    y_test,
    test_transform
)

# =====================================================
# HANDLE CLASS IMBALANCE
# =====================================================

print("\nClass Distribution")

class_counts = Counter(y_train)

for cls in sorted(class_counts.keys()):
    print(
        f"{class_names[cls]} : {class_counts[cls]}"
    )

weights = []

for label in y_train:
    weights.append(
        1.0 / class_counts[label]
    )

sampler = WeightedRandomSampler(
    weights,
    num_samples=len(weights),
    replacement=True
)

# =====================================================
# DATALOADER
# =====================================================

BATCH_SIZE = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)
# =====================================================
# MODEL
# =====================================================

print("\nLoading EfficientNet-B0...")

model = efficientnet_b0(
    weights="DEFAULT"
)

in_features = model.classifier[1].in_features

model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features, 7)
)

model = model.to(DEVICE)

# =====================================================
# LOSS
# =====================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=20
)

# =====================================================
# MIXED PRECISION
# =====================================================

scaler = torch.amp.GradScaler("cuda")

# =====================================================
# EARLY STOPPING
# =====================================================

BEST_ACC = 0

PATIENCE = 8

counter = 0

os.makedirs(
    "checkpoints",
    exist_ok=True
)

# =====================================================
# TRAINING HISTORY
# =====================================================

train_losses = []
val_accs = []

# =====================================================
# TRAIN LOOP
# =====================================================

EPOCHS = 30

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{EPOCHS}"
    )

    for images, labels in progress:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item()

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    scheduler.step()

    avg_loss = (
        running_loss /
        len(train_loader)
    )

    train_losses.append(avg_loss)

    # =================================================
    # VALIDATION
    # =================================================

    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(images)

            preds = torch.argmax(
                outputs,
                dim=1
            )

            predictions.extend(
                preds.cpu().numpy()
            )

            targets.extend(
                labels.numpy()
            )

    accuracy = accuracy_score(
        targets,
        predictions
    )

    val_accs.append(accuracy)

    print(
        f"\nEpoch {epoch+1}"
    )

    print(
        f"Train Loss : {avg_loss:.4f}"
    )

    print(
        f"Val Accuracy : {accuracy:.4f}"
    )

    # =================================================
    # SAVE BEST MODEL
    # =================================================

    if accuracy > BEST_ACC:

        BEST_ACC = accuracy

        torch.save(
            model.state_dict(),
            "checkpoints/best_model.pth"
        )

        counter = 0

        print(
            "Best Model Saved"
        )

    else:

        counter += 1

        print(
            f"No Improvement ({counter}/{PATIENCE})"
        )

    if counter >= PATIENCE:

        print(
            "\nEarly Stopping Triggered"
        )

        break

# =====================================================
# FINAL EVALUATION
# =====================================================

print("\nLoading Best Model...")

model.load_state_dict(
    torch.load(
        "checkpoints/best_model.pth"
    )
)

model.eval()

predictions = []
targets = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        preds = torch.argmax(
            outputs,
            dim=1
        )

        predictions.extend(
            preds.cpu().numpy()
        )

        targets.extend(
            labels.numpy()
        )

print("\nBest Accuracy:", BEST_ACC)

print("\nClassification Report\n")

print(
    classification_report(
        targets,
        predictions,
        target_names=class_names
    )
)

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    targets,
    predictions
)

plt.figure(figsize=(8,6))

plt.imshow(cm)

plt.title("Confusion Matrix")

plt.colorbar()

plt.xticks(
    range(7),
    class_names,
    rotation=45
)

plt.yticks(
    range(7),
    class_names
)

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png"
)

print(
    "\nConfusion matrix saved."
)

# =====================================================
# TRAINING CURVE
# =====================================================

plt.figure(figsize=(8,5))

plt.plot(
    train_losses,
    label="Train Loss"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "training_curve.png"
)

print(
    "Training curve saved."
)

print(
    "\nTraining Complete."
)