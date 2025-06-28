# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import json
import os

def train_model():
    print("Starting model training...")

    # Load IRIS dataset from CSV inside the 'data' folder
    csv_path = os.path.join('data', 'iris.csv')
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: '{csv_path}' not found. Please ensure the CSV file is in the 'data' directory.")
        return

    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = df['species'] # Target column is 'species'

    # Convert species names to numerical labels
    unique_species = y.unique()
    species_mapping = {name: i for i, name in enumerate(unique_species)}
    y_encoded = y.map(species_mapping)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # Initialize and train model
    model = LogisticRegression(max_iter=200, solver='liblinear')
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    # Save model
    model_path = 'iris_model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Save metrics
    metrics_path = 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}:")
    print(json.dumps(metrics, indent=4))

    print("Model training completed.")

if __name__ == "__main__":
    train_model()