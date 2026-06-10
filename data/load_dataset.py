# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import os
# pyrefly: ignore [missing-import]
import kagglehub
# pyrefly: ignore [missing-import]
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = "WineQT.csv"

# Load the latest version
df = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "yasserh/wine-quality-dataset",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

print("First 5 records:", df.head())

# Ensure raw data directory exists
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
output_dir = os.path.join(project_root, 'data', 'raw')
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, 'wine_quality.csv')
df.to_csv(output_path, index=False)
print(f"Dataset successfully saved to {output_path}")