import os
print("Current working directory:", os.getcwd())
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================
# LOAD DATA
# ==========================

data = pd.read_csv("truck_features.csv")

X = data.drop("label", axis=1)
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================
# RANDOM FOREST
# ==========================

rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_test)

print("\n===== RANDOM FOREST RESULTS =====")
print("Accuracy:", accuracy_score(y_test, rf_preds))
print("\nClassification Report:\n")
print(classification_report(y_test, rf_preds))

rf_cm = confusion_matrix(y_test, rf_preds)

plt.figure()
sns.heatmap(rf_cm, annot=True, fmt="d",
            xticklabels=["NT", "OT"],
            yticklabels=["NT", "OT"])
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("rf_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nRandom Forest Feature Importance:")
for feature, importance in zip(X.columns, rf_model.feature_importances_):
    print(f"{feature}: {importance:.4f}")

# ==========================
# DECISION TREE
# ==========================

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

dt_preds = dt_model.predict(X_test)

print("\n===== DECISION TREE RESULTS =====")
print("Accuracy:", accuracy_score(y_test, dt_preds))
print("\nClassification Report:\n")
print(classification_report(y_test, dt_preds))

dt_cm = confusion_matrix(y_test, dt_preds)

plt.figure()
sns.heatmap(dt_cm, annot=True, fmt="d",
            xticklabels=["NT", "OT"],
            yticklabels=["NT", "OT"])
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("dt_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nDecision Tree Feature Importance:")
for feature, importance in zip(X.columns, dt_model.feature_importances_):
    print(f"{feature}: {importance:.4f}")

# ==========================
# DECISION TREE VISUALIZATION
# ==========================
# ==========================
# DECISION TREE VISUALIZATION
# ==========================

import os

print("Saving Decision Tree Image...")

plt.figure(figsize=(18, 12))

plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=["NT", "OT"],
    filled=True,
    rounded=True,
    fontsize=10
)

file_path = os.path.join(os.getcwd(), "decision_tree.png")

plt.savefig("C:/Users/Sidhant/OneDrive/Desktop/Project_AoAI/decision_tree.png",
            dpi=300,
            bbox_inches="tight")

print("Decision Tree saved at:", file_path)

plt.close()   