import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# --- FUNGSI LOAD DATA (FIXED PATH) ---
def load_data():
    # Mengarah ke folder preprocessing di dalam folder yang sama dengan script
    path = "heart_preprocessing"
    
    # Cek apakah folder tersebut ada
    if not os.path.exists(path):
        # Fallback ke folder saat ini jika folder tidak ditemukan
        path = "." 

    print(f"[INFO] Membaca data dari folder: {path}")
    
    try:
        X_train = pd.read_csv(os.path.join(path, "X_train.csv"))
        y_train = pd.read_csv(os.path.join(path, "y_train.csv")).values.ravel()
        X_test = pd.read_csv(os.path.join(path, "X_test.csv"))
        y_test = pd.read_csv(os.path.join(path, "y_test.csv")).values.ravel()
        return X_train, y_train, X_test, y_test
    except FileNotFoundError as e:
        print(f"[ERROR] File data tidak ditemukan! Pastikan folder data ada di dalam MLProject. {e}")
        raise e

def main():
    # 1. Aktifkan Autolog sesuai syarat Kriteria 2
    # Autolog akan mencatat parameter, metrics, dan model secara otomatis.
    mlflow.autolog()
    
    # 2. Load Data
    X_train, y_train, X_test, y_test = load_data()

    # 3. Training Model
    # JANGAN gunakan set_experiment() atau start_run() agar tidak konflik dengan CI
    print("[INFO] Memulai training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Evaluasi Sederhana
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[SUCCESS] Training selesai. Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
