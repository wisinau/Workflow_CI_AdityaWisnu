import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

def load_data():
    # Menggunakan path relatif sesuai struktur MLProject
    path = "heart_failure_preprocessing"
    if not os.path.exists(path):
        path = "." # Fallback ke root jika folder tidak ditemukan
        
    X_train = pd.read_csv(os.path.join(path, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(path, "y_train.csv")).values.ravel()
    X_test = pd.read_csv(os.path.join(path, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(path, "y_test.csv")).values.ravel()
    return X_train, y_train, X_test, y_test

def main():
    # WAJIB: Aktifkan Autolog sesuai koreksi Kriteria 2
    mlflow.autolog()
    
    # JANGAN gunakan set_experiment atau start_run di sini agar tidak konflik
    X_train, y_train, X_test, y_test = load_data()

    print("[INFO] Training Model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"[SUCCESS] Accuracy: {accuracy_score(y_test, y_pred)}")

if __name__ == "__main__":
    main()
