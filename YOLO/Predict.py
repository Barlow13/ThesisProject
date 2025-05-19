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

# Load a pretrained YOLO11n model
model = YOLO(f"./runs/detect/{NICKNAME}/weights/best.pt")

# Define source as YouTube video URL
source = "IMG_9365.MP4"

# Run inference on the source
results = model(source, save=True, imgsz=TRAIN_IMG_SIZE, device=0)  # generator of Results objects