import pandas as pd
import mlflow
import mlflow.sklearn
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mlflow.set_tracking_uri(
    "https://dagshub.com/selenahans/Eksperimen-SML-Selena.mlflow"
)
mlflow.set_experiment("Student_Performance_Tuning")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Student_Performance_Preprocessed.csv")

df = pd.read_csv(DATA_PATH)
df = df.fillna(0)

X = df.drop(columns=["Performance Index"])
y = df["Performance Index"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_grid = {
    "n_estimators": [100, 150],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

base_model = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=3,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

with mlflow.start_run(run_name="RandomForest_Hyperparameter_Tuning"):

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2_Score", r2)

    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": best_model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    feature_importance.to_csv("feature_importance.csv", index=False)
    mlflow.log_artifact("feature_importance.csv")

    residuals = y_test - y_pred

    plt.figure()
    plt.scatter(y_pred, residuals)
    plt.axhline(0)
    plt.xlabel("Predicted Value")
    plt.ylabel("Residual")
    plt.title("Residual Plot (Tuned Model)")
    plt.savefig("residual_plot_tuning.png")
    plt.close()

    mlflow.log_artifact("residual_plot_tuning.png")

    mlflow.sklearn.log_model(
        best_model,
        "best_random_forest_tuned_model"
    )

print("Hyperparameter tuning selesai dan tercatat di MLflow (DagsHub)")
