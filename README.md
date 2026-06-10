# Wine Quality Prediction

An end-to-end Machine Learning Operations (MLOps) pipeline for predicting wine quality based on physicochemical tests. This project features a complete ETL process, model training using a Random Forest Classifier, and a Flask API for serving predictions.

## Project Structure

```text
.
├── data/
│   ├── raw/                  # Raw downloaded dataset
│   ├── processed/            # Preprocessed train/test splits and scaler
│   ├── load_dataset.py       # Script to fetch dataset from Kaggle
│   └── preprocess.py         # Script to clean, scale, and split data
├── model/
│   ├── train.py              # Script to train the Random Forest model
│   ├── evaluate.py           # Script to evaluate model performance
│   └── model.joblib          # Saved model artifact
├── app.py                    # Flask API for serving the model
├── requirement.txt           # Python dependencies
└── README.md                 # Project documentation
```

## Prerequisites

- Python 3.9+
- A virtual environment is highly recommended.

## Implementation Guide

### 1. Installation

First, navigate to the repository, create a virtual environment, and install the required packages:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirement.txt
```

### 2. Running the ML Pipeline

The pipeline is split into distinct steps. You can run them sequentially from the root of the project:

**Step A: Download Data**  
Fetches the `yasserh/wine-quality-dataset` using the Kaggle Hub API and saves it to `data/raw/`.
```bash
python data/load_dataset.py
```

**Step B: Preprocess Data**  
Cleans the data, applies `StandardScaler`, splits it into 80/20 train/test sets, and saves a `scaler.joblib` artifact for future inference.
```bash
python data/preprocess.py
```

**Step C: Train the Model**  
Trains a Random Forest Classifier on the processed data and saves the model to `model/model.joblib`.
```bash
python model/train.py
```

**Step D: Evaluate the Model**  
Tests the model on the unseen test dataset and prints the Accuracy, Classification Report, and Confusion Matrix.
```bash
python model/evaluate.py
```

### 3. Serving the Model via API

Once the model and scaler are generated, you can launch the Flask API to serve real-time predictions.

```bash
python app.py
```
The server will start at `http://127.0.0.1:5000`.

### 4. Testing the API

With the Flask server running, open a new terminal and send a POST request with sample physicochemical properties.

**cURL / bash:**
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "fixed acidity": 7.4,
    "volatile acidity": 0.70,
    "citric acid": 0.00,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11.0,
    "total sulfur dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
  }'
```

**Windows PowerShell:**
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"fixed acidity": 7.4, "volatile acidity": 0.70, "citric acid": 0.00, "residual sugar": 1.9, "chlorides": 0.076, "free sulfur dioxide": 11.0, "total sulfur dioxide": 34.0, "density": 0.9978, "pH": 3.51, "sulphates": 0.56, "alcohol": 9.4}'
```

**Expected Response:**
```json
{
  "predictions": [
    5
  ]
}
```

## Data Version Control (DVC)

This project has dependencies to support Data Version Control (`dvc`). Once your pipeline is tested, you can initialize DVC to track the `data/` directory and large model artifacts, preventing them from bloating your git repository.
