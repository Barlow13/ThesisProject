from ultralytics import YOLO

# ----- CONFIGURATION -----
NICKNAME = "yolo11n_person_cat_dog"


# Load the YOLO11 model
model = YOLO(f"./runs/detect/{NICKNAME}/weights/best.pt")

if __name__ == "__main__":
    # Validate the model
    metrics = model.val()  # no arguments needed, dataset and settings remembered
    print("Validation completed.")
    
    model.export(format="tflite", imgsz=64)
    
        
