import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(test_path, model_path):
    print(f"Loading test data from {test_path} and model from {model_path}")
    test_df = pd.read_csv(test_path)
    model = joblib.load(model_path)
    
    X_test = test_df.drop(columns=['quality'])
    y_test = test_df['quality']
    
    print("Evaluating model...")
    predictions = model.predict(X_test)
    
    acc = accuracy_score(y_test, predictions)
    print(f"Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

import os

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    test_path = os.path.join(project_root, 'data', 'processed', 'test.csv')
    model_path = os.path.join(project_root, 'model', 'model.joblib')
    
    evaluate_model(test_path, model_path)
