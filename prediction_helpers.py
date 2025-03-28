import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
import random
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


def evaluate_model(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return r2, rmse

def iterate_over_proportion(data, proportion, models):
    num_columns = max(1, int(data.shape[1] * proportion)) 
    random.seed(35)
    selected_columns = random.sample(data.columns.tolist(), num_columns)

    results = {}

    for model in models:
        model_name = model.__class__.__name__
        r2_scores = []
        rmse_scores = []

        results_list = Parallel(n_jobs=-1)(
            delayed(lambda col: evaluate_model(model, data.drop(columns=[col]), data[col]))(col)
            for col in selected_columns
        )

        r2_scores = [r2 for r2, _ in results_list]
        rmse_scores = [rmse for _, rmse in results_list]

        results[model_name] = {
            'Average R2': np.mean(r2_scores),
            'Average RMSE': np.mean(rmse_scores),
            'r2 array': sorted(r2_scores, reverse=True)
        }

    return results

def print_results(results):
    for model_name, metrics in results.items():
        print(f"\nModel: {model_name}")
        print(f"Average R²:   {metrics['Average R2']:.4f}")
        print(f"Average RMSE: {metrics['Average RMSE']:.4f}")


def plot_results(results):
    for model_name, metrics in results.items():
        plt.figure(figsize=(8, 4))
        plt.bar(range(len(metrics['r2 array'])), metrics['r2 array'], color='skyblue')
        plt.title(f"R² Values for {model_name}")
        plt.xlabel("Genes")
        plt.ylabel("R² Score")
        plt.ylim(0, 1)
        plt.show()

