import cv2
import numpy as np
import pandas as pd
import joblib
from ultralytics import YOLO
from collections import deque

# ==========================
# LOAD MODELS
# ==========================

yolo_model = YOLO("yolov8n.pt")
rf_model = joblib.load("rf_overload_model.pkl")
svm_model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

# ==========================
# SMOOTHING BUFFER
# ==========================

history = deque(maxlen=5)

# ==========================
# FEATURE EXTRACTION
# ==========================

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    area = width * height
    area_ratio = area / (width + height)
    aspect_ratio = width / height
    perimeter = 2 * (width + height)
    mean_intensity = gray.mean()

    edges = cv2.Canny(gray, 100, 200)
    edge_density = edges.mean()

    features = pd.DataFrame([{
        "width": width,
        "height": height,
        "area": area,
        "area_ratio": area_ratio,
        "aspect_ratio": aspect_ratio,
        "perimeter": perimeter,
        "mean_intensity": mean_intensity,
        "edge_density": edge_density
    }])

    return features


# ==========================
# VEHICLE TYPE HELPER
# ==========================

def get_vehicle_type(label):
    if label == "truck":
        return "Truck"
    elif label == "bus":
        return "Heavy Vehicle"
    elif label == "car":
        return "Car"
    else:
        return label


# ==========================
# MAIN SYSTEM
# ==========================

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame)

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        confidences = r.boxes.conf.cpu().numpy()

        for box, cls, conf in zip(boxes, classes, confidences):

            # Filter weak detections
            if conf < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box)
            label = yolo_model.names[int(cls)]

            # Process only vehicles
            if label in ["car", "truck", "bus"]:

                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                vehicle_name = get_vehicle_type(label)

                # ==========================
                # LOAD VEHICLE CHECK
                # ==========================
                if label in ["truck", "bus"]:

                    # Remove small detections
                    if (x2 - x1) < 100 or (y2 - y1) < 100:
                        continue

                    features = extract_features(crop)

                    rf_pred = rf_model.predict(features)[0]
                    features_scaled = scaler.transform(features)
                    svm_pred = svm_model.predict(features_scaled)[0]

                    # Decision logic
                    if rf_pred == svm_pred:
                        final = rf_pred
                    else:
                        final = svm_pred  # prefer SVM in conflict

                    # Smooth prediction
                    history.append(final)
                    final = max(set(history), key=history.count)

                    # Label + Color
                    if final == "OT":
                        text = f"{vehicle_name} | THREAT ({conf:.2f})"
                        color = (0, 0, 255)  # RED
                    else:
                        text = f"{vehicle_name} | SAFE ({conf:.2f})"
                        color = (0, 255, 0)  # GREEN

                    thickness = 3 if final == "OT" else 2

                else:
                    # Non-load vehicles
                    text = f"{vehicle_name} ({conf:.2f})"
                    color = (255, 255, 0)  # YELLOW
                    thickness = 2

                # ==========================
                # DRAW
                # ==========================
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)

    cv2.imshow("Real-Time AI System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()