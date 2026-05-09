import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Script started...")

# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv("truck_features.csv")

X = df.drop("label", axis=1)
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================
# RANDOM FOREST
# ==========================

print("\n===== RANDOM FOREST =====")

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, rf_preds))
print("\nClassification Report:\n")
print(classification_report(y_test, rf_preds))

rf_cm = confusion_matrix(y_test, rf_preds)
print("Confusion Matrix:\n", rf_cm)

# Save RF confusion matrix
plt.figure()
sns.heatmap(rf_cm, annot=True, fmt="d",
            xticklabels=["NT", "OT"],
            yticklabels=["NT", "OT"])
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("rf_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

# Save trained RF model
joblib.dump(rf_model, "rf_overload_model.pkl")
print("Random Forest model saved.")

# ==========================
# DECISION TREE
# ==========================

print("\n===== DECISION TREE =====")

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

dt_preds = dt_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, dt_preds))
print("\nClassification Report:\n")
print(classification_report(y_test, dt_preds))

dt_cm = confusion_matrix(y_test, dt_preds)
print("Confusion Matrix:\n", dt_cm)

# Save DT confusion matrix
plt.figure()
sns.heatmap(dt_cm, annot=True, fmt="d",
            xticklabels=["NT", "OT"],
            yticklabels=["NT", "OT"])
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("dt_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

# Feature importance
print("\nDecision Tree Feature Importance:")
for feature, importance in zip(X.columns, dt_model.feature_importances_):
    print(f"{feature}: {importance:.4f}")

# Save Decision Tree diagram
plt.figure(figsize=(18, 12))
plot_tree(dt_model,
          feature_names=X.columns,
          class_names=["NT", "OT"],
          filled=True,
          rounded=True)

plt.title("Decision Tree Visualization")
plt.savefig("decision_tree.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nAll outputs saved successfully.")