import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
import skops.io as sio


# Load dataset

os.makedirs("Data", exist_ok=True)  # ensure Data folder exists
os.makedirs("Results", exist_ok=True)
os.makedirs("Model", exist_ok=True)

df = pd.read_csv("Data/synthetic_heart_disease_dataset.csv")
df = df.sample(frac=1)  # shuffle


# Separate features & target

target = "Heart_Disease"
X = df.drop(columns=[target])
y = df[target]

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns


# Preprocessing pipelines

numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# Model pipeline

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
])


# Train/test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train model

model.fit(X_train, y_train)


# Evaluate

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
print("Accuracy:", accuracy)
print("F1 score:", f1)
print("\nClassification Report:\n")
print(classification_report(y_test, predictions))


# Save matrix
with open("Results/metrics.txt", "w") as outfile:
    outfile.write(f"Accuracy = {accuracy:.2f}, F1 Score = {f1:.2f}.\n")


# Confusion matrix

cm = confusion_matrix(y_test, predictions, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.savefig("Results/model_results.png", dpi=120)
# Save model
joblib.dump(model, "Model/heart_disease_model.pkl")
print("Model saved successfull!")
