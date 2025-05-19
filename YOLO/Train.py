import os
from ultralytics import YOLO
import torch

# ----- CONFIGURATION -----
EPOCHS = 16
BATCH_SIZE = 32
TRAIN_IMG_SIZE = 224
WORKERS = 24
IMG_HEIGHT, IMG_WIDTH = 64, 64
EXPORT_IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
CLASSES = ["person", "cat", "dog"]
CLASS_INDICES = [0, 15, 16]
NICKNAME = "yolo11n_person_cat_dog"
RESUME = False


# Determine the best available device
if torch.cuda.is_available():
    DEVICE = 0  # NVIDIA GPU
else:
    DEVICE = "cpu"  # CPU fallback

# Load a pretrained model
model = YOLO("yolo11n.pt")

# ----- TRAINING PHASE -----
if __name__ == "__main__":
    print(f"Starting training on classes: {CLASSES}")
    print(f"Using batch size: {BATCH_SIZE}, image size: {TRAIN_IMG_SIZE}")
    print(f"CPU workers: {WORKERS}, Device: {DEVICE}")

    # training configuration
    Train = model.train(
        # Training/Model settings
        data = "COCOTRAIN.yaml",
        epochs = EPOCHS,
        patience = int(EPOCHS * .25),
        batch = BATCH_SIZE,
        imgsz = TRAIN_IMG_SIZE,
        save = True,
        save_period = int(EPOCHS * .125),
        cache = "disk",
        device = DEVICE,
        workers = WORKERS,
        name = NICKNAME,
        seed = 42,
        classes = CLASS_INDICES,
        multi_scale = True,
        cos_lr = True,
        resume= RESUME,
        amp = True,
        lr0 = 0.01,
        lrf = 0.0001,
        warmup_epochs = int(EPOCHS * .125),
        plots=True,
        mosaic = 0.0,
        mixup = 0.2,
        degrees = 10.0,
        translate = 0.1,
        scale = 0.5,
        fliplr = 0.5,
    )

    print("Training completed.")

    