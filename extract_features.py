import os
import cv2
import numpy as np
import pandas as pd

dataset_path = "dataset"
classes = ["OT", "NT"]

data = []

for label in classes:
    folder_path = os.path.join(dataset_path, label)

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (600, 400))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(blur, 50, 150)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            continue

        largest_contour = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(largest_contour)

        area = cv2.contourArea(largest_contour)
        bbox_area = w * h

        if bbox_area == 0:
            continue

        area_ratio = area / bbox_area
        aspect_ratio = w / h
        perimeter = cv2.arcLength(largest_contour, True)
        mean_intensity = np.mean(gray)

        # NEW FEATURE
        edge_density = edges.mean()

        data.append([
            w,
            h,
            area,
            area_ratio,
            aspect_ratio,
            perimeter,
            mean_intensity,
            edge_density,
            label
        ])

df = pd.DataFrame(data, columns=[
    "width",
    "height",
    "area",
    "area_ratio",
    "aspect_ratio",
    "perimeter",
    "mean_intensity",
    "edge_density",
    "label"
])

df.to_csv("truck_features.csv", index=False)

print("Feature extraction complete.")