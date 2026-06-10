import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_path, output_dir):
    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    
    # Typically in this dataset the target is 'quality'
    # Drop 'Id' column if it exists
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])
    
    # Check for missing values and fill/drop if necessary
    if df.isnull().sum().any():
        print("Handling missing values...")
        df = df.dropna() # Simple approach, could also impute

    # Split features and target
    X = df.drop(columns=['quality'])
    y = df['quality']
    
    # Split train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    import joblib
    scaler_path = os.path.join(output_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # Re-attach the target column for saving
    train_df = X_train_scaled.copy()
    train_df['quality'] = y_train.values
    
    test_df = X_test_scaled.copy()
    test_df['quality'] = y_test.values
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Preprocessing complete. Train and test sets saved to {output_dir}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    input_path = os.path.join(project_root, 'data', 'raw', 'wine_quality.csv')
    output_dir = os.path.join(project_root, 'data', 'processed')
    
    preprocess_data(input_path, output_dir)
