import cv2
import numpy as np
import pandas as pd
import joblib
from ultralytics import YOLO

# Load models
yolo_model = YOLO("yolov8n.pt")
rf_model = joblib.load("rf_overload_model.pkl")

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    area = width * height
    area_ratio = area / (width + height)
    aspect_ratio = width / height
    perimeter = 2 * (width + height)
    mean_intensity = gray.mean()

    features = pd.DataFrame([{
        "width": width,
        "height": height,
        "area": area,
        "area_ratio": area_ratio,
        "aspect_ratio": aspect_ratio,
        "perimeter": perimeter,
        "mean_intensity": mean_intensity
    }])

    return features


# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame)

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for box, cls in zip(boxes, classes):

            x1, y1, x2, y2 = map(int, box)
            label = yolo_model.names[int(cls)]

            if label in ["truck", "bus", "car"]:

                truck_crop = frame[y1:y2, x1:x2]

                if truck_crop.size == 0:
                    continue

                features = extract_features(truck_crop)
                prediction = rf_model.predict(features)[0]

                text = f"{label} | {prediction}"

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,text,(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,(0,255,0),2)

    cv2.imshow("Real-Time AI System", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()