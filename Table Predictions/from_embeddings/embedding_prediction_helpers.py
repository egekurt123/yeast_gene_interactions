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
        X_train, X_test, y_train, y_test = gene_based_train_test_split(embeddings, prediction_type='dmf', random_state=42)
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

    return r2


def run_PCA(X, plot=True):
    N=128
    pca = PCA(n_components=N)
    X_PCA = pca.fit_transform(X.copy())
    ev = pca.explained_variance_ratio_.cumsum()
    if plot:
        print(ev[-1])
        plt.plot(range(0, N), ev)
    return X_PCA


def run_Ridge_Regression(X=None , y=None, color="blue", plot=True, gene_holdout=False, embeddings=None, alpha=10.0):
    random.seed(38)

    if gene_holdout:
        X_train, X_test, y_train, y_test = gene_based_train_test_split(embeddings, prediction_type='dmf', random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ridge_model = Ridge(alpha=alpha, random_state=42).fit(X_train, y_train)
    y_pred = ridge_model.predict(X_test)
 
    print("Ridge Regression")

    r2 = ridge_model.score(X_test, y_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"R2: {r2:.3f}")
    print(f"RMSE: {rmse:.3f}")

    if plot:
        plt.scatter(y_pred, y_test, alpha=0.5, c=color)
        plt.xlabel("y-pred", fontsize=12)
        plt.ylabel("y-true", fontsize=12)
        plt.title("Ridge Regression Predictions", fontweight='bold', fontsize=14, pad=10)
        plt.show()
    
    return r2


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

    return r2


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

    return r2


def predict_all_models(embeddings, combination, prediction_type='gi_score'):

    print("Running for combination:", combination)
    print("\n")

    X,y = preprocess_data(embeddings, prediction_type=prediction_type, subfolder=False)

    r2_linear = run_Linear_Regression(X, y, "darkblue", plot=False, pca=False)
    print("\n")

    X_PCA = run_PCA(X, plot=False)

    r2_linear_pca= run_Linear_Regression(X_PCA, y, "darkred", plot=False, pca=True)
    print("\n")

    r2_ridge = run_Ridge_Regression(X, y, "darkred", plot=False, alpha=200.0)
    print("\n")

    r2_xgboost = run_XGBoost(X, y, plot=False)
    print("\n")

    r2_randomforest = run_Random_Forest(X, y, embeddings, plot=False)
    print("\n")

    return [r2_linear_pca, r2_ridge, r2_xgboost, r2_randomforest]

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


def gene_based_train_test_split(embeddings, prediction_type='dmf', holdout_fraction=0.2, random_state=42):

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

    return X_train, X_test, y_train, y_test


def plot_r2_combinations(dnalm_comb=[], dnalm_yeastnet_comb=[], dnalm_genex_comb=[], dnalm_yeastnet_genex_comb=[]):

    datasets = ['DNALM', 'DNALM + YeastNet', 'DNALM + Gene Expr', 'DNALM + YeastNet\n+ Gene Expressions']
    models = ['Linear Regression with PCA', 'Ridge Regression', 'XGBoost Regression', 'Random Forest Regression']

    r2_data = {
        'DNALM': dnalm_comb,
        'DNALM + YeastNet': dnalm_yeastnet_comb,
        'DNALM + Gene Expr': dnalm_genex_comb,
        'DNALM + YeastNet\n+ Gene Expressions': dnalm_yeastnet_genex_comb
    }

    colors = ['skyblue', 'lightgreen', 'orange', 'lightcoral']

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    x = np.arange(len(datasets))
    width = 0.2

    for i, model in enumerate(models):
        model_scores = [r2_data[dataset][i] for dataset in datasets]
        bars = ax.bar(x + i*width, model_scores, width, label=model, color=colors[i], alpha=0.8)

        for bar, score in zip(bars, model_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                     f'{score:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('R² Scores Across Dataset Combinations', fontweight='bold', fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 0.6)

    plt.tight_layout()
    plt.show()


def predict_all_models_holdout(embeddings, combination, prediction_type='dmf'):

    print("Running for combination:", combination)
    print("\n")

    r2_linear = run_Linear_Regression(embeddings=embeddings, gene_holdout=True, color='darkblue', plot=False)
    print("\n")

    r2_ridge = run_Ridge_Regression(embeddings=embeddings, gene_holdout=True, color='darkblue', alpha=200.0, plot=False)
    print("\n")

    r2_xgboost = run_XGBoost(embeddings=embeddings, gene_holdout=True, plot=False)
    print("\n")

    r2_randomforest = run_Random_Forest(embeddings=embeddings, gene_holdout=True, plot=False)
    print("\n")

    return [0, r2_ridge, r2_xgboost, r2_randomforest]
