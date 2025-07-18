import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import random
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def create_neural_network(input_dim):
    model = Sequential([
    Input(shape=(input_dim,)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])

    model.compile(
    optimizer=Adam(learning_rate=0.001), 
    loss='mse',
    metrics=['mse']
)           
    return model

def evaluate_model(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if isinstance(model, Sequential):  # Neural network case
        input_dim = X_train.shape[1]
        model = create_neural_network(input_dim)
        
        model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=32,
            verbose=0
        )
        
        y_pred = model.predict(X_test, verbose=0).flatten()
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return r2, rmse

def iterate_over_proportion_only_interactions(data, proportion, models):
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
            'r2 array': r2_scores,
            'gene array': selected_columns

        }

    return results

def iterate_over_proportion_only_embeddings(data, embeddings, proportion, models):

    if 'gene_id' not in data.columns:
        data = data.reset_index()
    if 'gene_id' not in embeddings.columns:
        embeddings = embeddings.reset_index()

    shared_ids = set(data['gene_id']) & set(embeddings['gene_id'])
    data_shared = data[data['gene_id'].isin(shared_ids)].sort_values('gene_id')
    embeddings_shared = embeddings[embeddings['gene_id'].isin(shared_ids)].sort_values('gene_id')

    target_cols = [col for col in data_shared.columns if col != 'gene_id']
    num_columns = max(1, int(len(target_cols) * proportion))
    random.seed(35)
    selected_columns = random.sample(target_cols, num_columns)

    results = {}
    X = embeddings_shared.drop(columns=['gene_id'])

    for model in models:
        model_name = model.__class__.__name__
        results_list = Parallel(n_jobs=-1)(
            delayed(lambda col: evaluate_model(model, X, data_shared[col]))(col)
            for col in selected_columns
        )
        r2_scores = [r2 for r2, _ in results_list]
        rmse_scores = [rmse for _, rmse in results_list]
        results[model_name] = {
            'Average R2': np.mean(r2_scores),
            'Average RMSE': np.mean(rmse_scores),
            'r2 array': r2_scores,
            'gene array': selected_columns
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
        plt.bar(range(len(metrics['r2 array'])), sorted(metrics['r2 array'], reverse=True), color='skyblue')
        plt.title(f"R² Values for {model_name}")
        plt.xlabel("Genes")
        plt.ylabel("R² Score")
        plt.ylim(0, 1)
        plt.show()

def print_top_genes(results, model_name, top_n=10):
    if model_name not in results:
        print(f"Model '{model_name}' not found in results.")
        return
    
    r2_scores = results[model_name]['r2 array']
    average_r2 = np.mean(results[model_name]['r2 array'])

    genes = results[model_name]['gene array']

    gene_r2_df = pd.DataFrame({'Gene': genes, 'R2 Score': r2_scores})
    top_genes = gene_r2_df.sort_values(by='R2 Score', ascending=False).head(top_n)

    print(f"\nTop {top_n} Genes by R² Score for {model_name}:")
    print(top_genes.to_string(index=False))

    return top_genes, average_r2