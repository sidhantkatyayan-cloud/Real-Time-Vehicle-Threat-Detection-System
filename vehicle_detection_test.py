import os
import cv2
from ultralytics import YOLO

# Load pretrained YOLOv8 nano model
model = YOLO("yolov8n.pt")  # will auto-download if not present

# Automatically pick first image from NT folder
dataset_folder = "dataset/NT"

image_files = [f for f in os.listdir(dataset_folder) if f.endswith((".jpg", ".png", ".jpeg"))]

if len(image_files) == 0:
    print("No images found in NT folder.")
    exit()

img_path = os.path.join(dataset_folder, image_files[0])
print("Testing image:", img_path)

# Run YOLO detection
results = model(img_path)

# Get result image with bounding boxes
result_img = results[0].plot()

# Convert for OpenCV display
result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)

# Show image
cv2.imshow("YOLO Vehicle Detection", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()