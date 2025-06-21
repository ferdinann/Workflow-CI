import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import os
from dotenv import load_dotenv

def modeling_with_tuning(X_train_path, X_test_path, y_train_path, y_test_path):
    # Load data hasil preprocessing & split
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

if __name__ == "__main__":
    # Path dataset hasil split
    X_train_path = "Dataset Preprocessing/X_train.csv"
    X_test_path = "Dataset Preprocessing/X_test.csv"
    y_train_path = "Dataset Preprocessing/y_train.csv"
    y_test_path = "Dataset Preprocessing/y_test.csv"

    # Autentikasi ke DagsHub
    load_dotenv()
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    if not username or not password:
        raise EnvironmentError("MLFLOW_TRACKING_USERNAME dan MLFLOW_TRACKING_PASSWORD harus di-set sebagai environment variable")

    # Set MLflow tracking URI
    mlflow.set_tracking_uri("https://dagshub.com/ferdinantag8/Membangun_model.mlflow/")
    mlflow.set_experiment("Healthcare-Diabetes")

    with mlflow.start_run(run_name="Modelling_tunning_manuallog"):
        model, accuracy, report, best_params, X_test = modeling_with_tuning(X_train_path, X_test_path, y_train_path, y_test_path)

        # Log params
        for param, value in best_params.items():
            mlflow.log_param(param, value)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", report["weighted avg"]["precision"])
        mlflow.log_metric("recall", report["weighted avg"]["recall"])
        mlflow.log_metric("f1_score", report["weighted avg"]["f1-score"])

        # Set tag untuk menandai tahap lifecycle model
        mlflow.set_tag("stage", "tunning")
        mlflow.set_tag("model_type", "RandomForestClassifier")

        # Tambah signature
        input_example = X_test.iloc[:1]
        signature = infer_signature(X_test, model.predict(X_test))

        # Simpan model ke dalam MLflow dengan nama artifact rf_model
        mlflow.sklearn.log_model(
            model,
            artifact_path="rf_best_model",
            signature=signature,
            input_example=input_example,
            conda_env="conda.yaml"
        )

        print("Proses tunning dan logged MLflow selesai")