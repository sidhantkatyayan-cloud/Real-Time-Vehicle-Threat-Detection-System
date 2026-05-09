import numpy as np
import pandas as pd

# Create synthetic dataset
np.random.seed(42)

data_size = 400

headlight_count = np.random.choice([1, 2], data_size)

headlight_distance = np.random.normal(30, 10, data_size)
intensity = np.random.normal(220, 20, data_size)
width = np.random.normal(50, 15, data_size)
height = np.random.normal(40, 10, data_size)
shape_irregularity = np.random.uniform(0, 0.5, data_size)

labels = []

for i in range(data_size):
    if headlight_count[i] == 1:
        labels.append("Bike")
    elif width[i] > 65:
        labels.append("Truck")
    else:
        labels.append("Car")

df = pd.DataFrame({
    "headlight_count": headlight_count,
    "headlight_distance": headlight_distance,
    "intensity": intensity,
    "width": width,
    "height": height,
    "shape_irregularity": shape_irregularity,
    "label": labels
})

print(df.head())

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Separate features and target
X = df.drop("label", axis=1)
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()