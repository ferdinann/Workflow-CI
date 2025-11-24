import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import mlflow
from mlflow.models.signature import infer_signature
import os
# from dotenv import load_dotenv # Opsional di CI

def modeling_with_tuning(X_train_path, X_test_path, y_train_path, y_test_path):
    # Load data hasil preprocessing & split
    # Pastikan file path benar sesuai struktur direktori saat script dijalankan
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path).squeeze()
    y_test = pd.read_csv(y_test_path).squeeze()

    # Menentukan hyperparameter grid
    param_grid = {
        "n_estimators": [100, 150],
        "max_depth": [10, 15],
        "min_samples_split": [2, 4]
    }

    # Inisialisasi model & GridSearch
    model = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        model, param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_

    # Prediksi & Evaluasi
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Hasil
    print("Akurasi:", accuracy)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return best_model, accuracy, report, grid_search.best_params_, X_test

# ... (kode impor di atas tetap sama)

if __name__ == "__main__":
    X_train_path = "Dataset Preprocessing/X_train.csv"
    X_test_path = "Dataset Preprocessing/X_test.csv"
    y_train_path = "Dataset Preprocessing/y_train.csv"
    y_test_path = "Dataset Preprocessing/y_test.csv"

    print(f"Tracking URI: {mlflow.get_tracking_uri()}")

    # Logika penentuan Run
    if mlflow.active_run():
        run = mlflow.active_run()
        print(f"Active run detected: {run.info.run_id}")
    else:
        print("No active run detected. Starting new run manually.")
        mlflow.set_experiment("Healthcare-Diabetes")
        run = mlflow.start_run(run_name="Modelling_tuning_manuallog")

    # --- MULAI PROSES TRAINING ---
    with run:
        # Simpan Run ID ke file teks agar bisa dibaca CI/CD (PENTING!)
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)
        
        print(f"Run ID {run.info.run_id} saved to run_id.txt")

        # Jalankan fungsi modeling
        model, accuracy, report, best_params, X_test = modeling_with_tuning(X_train_path, X_test_path, y_train_path, y_test_path)

        # Log params & metrics
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", report["weighted avg"]["precision"])
        mlflow.log_metric("recall", report["weighted avg"]["recall"])
        mlflow.log_metric("f1_score", report["weighted avg"]["f1-score"])

        mlflow.set_tag("stage", "tuning")
        mlflow.set_tag("model_type", "RandomForestClassifier")

        input_example = X_test.iloc[:1]
        signature = infer_signature(X_test, model.predict(X_test))

        # Log Model
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            conda_env="conda.yaml"
        )
    
    print("Proses selesai")
