import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
import xgboost as xgb


def preprocess_data(embeddings, orthologous_groups = False, prediction_type = 'gi_score', subfolder=True):

    path = '../../extracted_data/interaction_table_all.csv'

    if subfolder:
        path = '../' + path

    interaction_table = pd.read_csv(path, sep=',', index_col=0)

    available_genes = set(embeddings.index)
    filtered = interaction_table[
        interaction_table['query_gene'].isin(available_genes) &
        interaction_table['array_gene'].isin(available_genes)
    ].copy()

    filtered_sample = filtered.sample(n=10000, random_state=42)

    X = np.array([
        np.concatenate([embeddings.loc[row['query_gene']].values,
                        embeddings.loc[row['array_gene']].values])
        for _, row in filtered_sample.iterrows()
    ])

    y = filtered_sample[prediction_type].values
    
    return X, y

def run_Linear_Regression(X, y, color, plot=True, pca=False):
    random.seed(38)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lm = LinearRegression().fit(X_train, y_train)
    y_pred = lm.predict(X_test)

    if pca:
        print("Linear Regression with PCA")
    else:
        print("Linear Regression")

    print('R2: ', lm.score(X_test, y_test))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

    if plot:
        plt.scatter(y_pred, y_test, alpha=0.5, c=color)
        plt.xlabel("y-pred", fontsize=12)
        plt.ylabel("y-true", fontsize=12)
        plt.title("Linear Regression Predictions", fontweight='bold', fontsize=14, pad=10)
        plt.show()


def run_PCA(X, plot=True):
    N=128
    pca = PCA(n_components=N)
    X_PCA = pca.fit_transform(X.copy())
    ev = pca.explained_variance_ratio_.cumsum()
    if plot:
        print(ev[-1])
        plt.plot(range(0, N), ev)
    return X_PCA


def run_XGBoost(X, y, embeddings=None, plot=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("XGBoost Regression")
    print(f"R2: {r2:.3f}")
    print(f"RMSE: {rmse:.3f}")

    if plot:
        emb_cols = embeddings.columns.tolist()
        feature_names = [f"query_{col}" for col in emb_cols] + [f"array_{col}" for col in emb_cols]
        
        importances = xgb_model.feature_importances_
        indices = np.argsort(importances)[-20:]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.title('XGBoost Feature Importance')
        plt.tight_layout()
        plt.show()


def run_Random_Forest(X, y, embeddings, plot=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("Random Forest Regression")
    print(f"Random Forest R² score: {r2:.3f}")
    print(f"Random Forest RMSE: {rmse:.3f}")


    if plot:
        # Add the column name
        emb_cols = embeddings.columns.tolist()
        n_emb_dims = len(emb_cols)
        feature_names = [f"query_{col}" for col in emb_cols] + [f"array_{col}" for col in emb_cols]
        #######################

        plt.figure(figsize=(10, 6))
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[-20:] 

        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.title('Random Forest Feature Importance')
        plt.tight_layout()
        plt.show()

def predict_all_models(embeddings, combination, prediction_type='gi_score'):

    print("Running for combination:", combination)
    print("\n")

    X,y = preprocess_data(embeddings, prediction_type=prediction_type, subfolder=False)

    run_Linear_Regression(X, y, "darkblue", plot=False, pca=False)
    print("\n")

    X_PCA = run_PCA(X, plot=False)

    run_Linear_Regression(X_PCA, y, "darkred", plot=False, pca=True)
    print("\n")


    run_XGBoost(X, y, plot=False)
    print("\n")

    run_Random_Forest(X, y, embeddings, plot=False)
    print("\n")

