import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

INPUT_PATH = "dataset_raw/Student_Performance_raw.csv"
OUTPUT_DIR = "dataset_preprocessing"
OUTPUT_PATH = f"{OUTPUT_DIR}/Student_Performance_Preprocessed.csv"

def preprocess_data():

    data = pd.read_csv(INPUT_PATH)

    if data.duplicated().sum() > 0:
        data = data.drop_duplicates()

    data['Extracurricular Activities'] = data['Extracurricular Activities'].map({
        'Yes': 1,
        'No': 0
    })

    numerical_features = [
        'Hours Studied',
        'Previous Scores',
        'Sleep Hours',
        'Sample Question Papers Practiced'
    ]

    scaler = StandardScaler()
    data[numerical_features] = scaler.fit_transform(
        data[numerical_features]
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)

    print("Preprocessing selesai.")
    print(f"Dataset tersimpan di: {OUTPUT_PATH}")

if __name__ == "__main__":
    preprocess_data()
