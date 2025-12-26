import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import os

# --- LOAD DATA ---
# Pastikan path ini sesuai dengan struktur folder Anda
DATA_PATH = "heart_preprocessing" 

def load_data():
    if not os.path.exists(DATA_PATH):
        print(f"[WARNING] Folder {DATA_PATH} tidak ditemukan, mencari di root...")
        path = "."
    else:
        path = DATA_PATH
        
    X_train = pd.read_csv(os.path.join(path, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(path, "y_train.csv")).values.ravel()
    X_test = pd.read_csv(os.path.join(path, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(path, "y_test.csv")).values.ravel()
    return X_train, y_train, X_test, y_test

def main():
    # 1. Aktifkan Autolog (INI KUNCINYA)
    mlflow.autolog()
    
    # Set Experiment (Boleh pakai nama bebas)
    mlflow.set_experiment("Heart_Failure_Autolog_Aditya")

    print("[INFO] Loading Data...")
    X_train, y_train, X_test, y_test = load_data()

    with mlflow.start_run():
        print("[INFO] Training Model...")
        # Model sederhana tanpa tuning rumit
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        print(f"Metrics: Acc={acc}, F1={f1}")
        print("[SUCCESS] Training Selesai! Log tersimpan otomatis oleh Autolog.")

if __name__ == "__main__":
    main()