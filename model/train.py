import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

def train_model(train_path, model_output_path):
    print(f"Loading training data from {train_path}")
    train_df = pd.read_csv(train_path)
    
    X_train = train_df.drop(columns=['quality'])
    y_train = train_df['quality']
    
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"Model saved to {model_output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    train_path = os.path.join(project_root, 'data', 'processed', 'train.csv')
    model_output_path = os.path.join(project_root, 'model', 'model.joblib')
    
    train_model(train_path, model_output_path)
