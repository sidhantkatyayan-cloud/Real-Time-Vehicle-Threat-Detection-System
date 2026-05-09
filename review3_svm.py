import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ==========================
# LOAD DATA
# ==========================

data = pd.read_csv("truck_features.csv")

X = data.drop("label", axis=1)
y = data["label"]

# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================
# FEATURE SCALING
# ==========================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================
# SVM MODEL
# ==========================

svm_model = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced')
svm_model.fit(X_train, y_train)

# ==========================
# TRAIN ACCURACY
# ==========================

train_preds = svm_model.predict(X_train)
train_acc = accuracy_score(y_train, train_preds)

# ==========================
# TEST ACCURACY
# ==========================

test_preds = svm_model.predict(X_test)
test_acc = accuracy_score(y_test, test_preds)

print("SVM Train Accuracy:", train_acc)
print("SVM Test Accuracy:", test_acc)

print("\nClassification Report:\n")
print(classification_report(y_test, test_preds))

# ==========================
# CONFUSION MATRIX
# ==========================

cm = confusion_matrix(y_test, test_preds)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=["NT", "OT"],
            yticklabels=["NT", "OT"])
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("svm_confusion_matrix.png")
plt.close()

# ==========================
# MODEL COMPARISON GRAPH
# ==========================

rf_accuracy = 0.80  # replace if needed

models = ["Random Forest", "SVM"]
accuracies = [rf_accuracy, test_acc]

plt.figure()
plt.bar(models, accuracies)
plt.title("Model Comparison")
plt.ylabel("Accuracy")
plt.savefig("model_comparison.png")
plt.close()

# ==========================
# SAVE MODEL + SCALER
# ==========================

joblib.dump(svm_model, "svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nSVM model and scaler saved successfully!")