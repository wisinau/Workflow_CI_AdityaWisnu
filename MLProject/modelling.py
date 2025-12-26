import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import os

# --- LOAD DATA ---
def load_data():
    # Cek berbagai kemungkinan lokasi data
    paths = [
        "heart_failure_preprocessing", 
        "../heart_failure_preprocessing", 
        "data"
    ]
    
    data_path = None
    for p in paths:
        if os.path.exists(p):
            data_path = p
            break
            
    if not data_path:
        print("[WARNING] Folder data tidak ditemukan di path standar.")
        # Fallback terakhir: coba cari di current directory
        data_path = "."
        
    print(f"[INFO] Menggunakan data dari: {data_path}")

    try:
        X_train = pd.read_csv(os.path.join(data_path, "X_train.csv"))
        y_train = pd.read_csv(os.path.join(data_path, "y_train.csv")).values.ravel()
        X_test = pd.read_csv(os.path.join(data_path, "X_test.csv"))
        y_test = pd.read_csv(os.path.join(data_path, "y_test.csv")).values.ravel()
        return X_train, y_train, X_test, y_test
    except Exception as e:
        print(f"[ERROR] Gagal load data: {e}")
        raise e

def main():
    # 1. Aktifkan Autolog
    mlflow.autolog()
    
    # --- BAGIAN INI DIHAPUS AGAR TIDAK ERROR DI GITHUB ACTIONS ---
    # mlflow.set_experiment("Heart_Failure_Autolog_Aditya") 
    
    print("[INFO] Loading Data...")
    X_train, y_train, X_test, y_test = load_data()

    # start_run() otomatis memakai run yang dibuat oleh 'mlflow run'
    with mlflow.start_run():
        print("[INFO] Training Model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        print("[SUCCESS] Training Selesai! Log tersimpan otomatis.")

if __name__ == "__main__":
    main()
