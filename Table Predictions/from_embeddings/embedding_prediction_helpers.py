import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, classification_report, confusion_matrix
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
import xgboost as xgb
import seaborn as sns


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


def preprocess_data_classification(embeddings, subfolder=True):
    X, y = preprocess_data(embeddings, prediction_type='gi_score', subfolder=subfolder)
    y = gi_score_to_class(y)
    return X, y
    



def gi_score_to_class(gi_scores, pos_thresh=0.08, neg_thresh=-0.08):
    """
    Map gi_scores to classes:
    - 1: positive (gi_score > pos_thresh)
    - -1: negative (gi_score < neg_thresh)
    - 0: neutral (otherwise)
    """
    classes = np.ones_like(gi_scores, dtype=int)
    classes[gi_scores > pos_thresh] = 0
    classes[gi_scores < neg_thresh] = 2
    return classes


def run_Linear_Regression(X=None , y=None, color="blue", plot=True, pca=False, gene_holdout=False, embeddings=None):
    random.seed(38)

    if gene_holdout:
        X_train, X_test, y_train, y_test = gene_based_train_test_split(embeddings, prediction_type='dmf', random_state=42, pca=True)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lm = LinearRegression().fit(X_train, y_train)
    y_pred = lm.predict(X_test)

    if pca:
        print("Linear Regression with PCA")
    else:
        print("Linear Regression")

    r2 = lm.score(X_test, y_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"R2: {r2:.3f}")
    print(f"RMSE: {rmse:.3f}")

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


def run_XGBoost(X=None , y=None, embeddings=None, plot=True, gene_holdout=False):

    if gene_holdout:
        X_train, X_test, y_train, y_test = gene_based_train_test_split(embeddings, prediction_type='dmf', random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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


def run_Random_Forest(X=None, y=None, embeddings=None, plot=True, gene_holdout=False):
    
    if gene_holdout:
        X_train, X_test, y_train, y_test = gene_based_train_test_split(embeddings, prediction_type='dmf', random_state=42)
    else:
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
    print(f"R2: {r2:.3f}")
    print(f"RMSE: {rmse:.3f}")


    if plot:
        emb_cols = embeddings.columns.tolist()
        n_emb_dims = len(emb_cols)
        feature_names = [f"query_{col}" for col in emb_cols] + [f"array_{col}" for col in emb_cols]

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


def run_RandomForest_Classifier(X, y, plot=True):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    if plot:
        import seaborn as sns
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Random Forest Confusion Matrix')
        plt.show()


def run_RandomForest_Classifier(X, y, plot=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    if plot:
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Random Forest Confusion Matrix')
        plt.show()

def run_XGBoost_Classifier(X, y, plot=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = xgb.XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42, eval_metric='mlogloss')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    if plot:
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('XGBoost Confusion Matrix')
        plt.show()


def gene_based_train_test_split(embeddings, prediction_type='dmf', holdout_fraction=0.2, random_state=42, pca=False):

    interaction_table = pd.read_csv('../../extracted_data/interaction_table_all.csv', sep=',', index_col=0)

    available_genes = set(embeddings.index)
    filtered = interaction_table[
        interaction_table['query_gene'].isin(available_genes) &
        interaction_table['array_gene'].isin(available_genes)
    ].copy()

    filtered = filtered.sample(n=10000, random_state=random_state)

    all_genes = pd.unique(filtered[['query_gene', 'array_gene']].values.ravel())
    rng = np.random.default_rng(random_state)
    rng.shuffle(all_genes)
    # Select genes until at least holdout_fraction of rows are covered
    selected = set()
    covered = np.zeros(len(filtered), dtype=bool)
    i = 0
    while covered.mean() < holdout_fraction and i < len(all_genes):
        selected.add(all_genes[i])
        covered = covered | (filtered['query_gene'].isin(selected) | filtered['array_gene'].isin(selected))
        i += 1
    holdout_genes = list(selected)
    print(f"Randomly selected {len(holdout_genes)} holdout genes to cover ~{holdout_fraction*100:.0f}% of data.")

    test_mask = filtered['query_gene'].isin(holdout_genes) | filtered['array_gene'].isin(holdout_genes)
    test_df = filtered[test_mask]
    train_df = filtered[~test_mask]

    def make_X_y(df):
        X = np.array([
            np.concatenate([embeddings.loc[row['query_gene']].values,
                            embeddings.loc[row['array_gene']].values])
            for _, row in df.iterrows()
        ])
        y = df[prediction_type].values
        return X, y

    X_train, y_train = make_X_y(train_df)
    X_test, y_test = make_X_y(test_df)

    if pca:
        X_train = run_PCA(X_train, plot=False)
        X_test = run_PCA(X_test, plot=False)

    return X_train, X_test, y_train, y_test
