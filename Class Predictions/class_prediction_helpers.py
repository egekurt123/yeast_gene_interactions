import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import sklearn.linear_model
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import PrecisionRecallDisplay
import matplotlib.pyplot as plt



classes = pd.read_csv('../extracted_data/essentiality_classes.csv', sep=',', index_col=0)
classes


def get_class_predictions(embeddings):

    dataset = embeddings.merge(classes, left_index=True, right_index=True)
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


    classifiers = [
    RandomForestClassifier(100), 
    LogisticRegression(max_iter=500)
    ]

    for classifier in classifiers:
        classifier.fit(X_train, y_train)
    
    fig, ax = plt.subplots()

    for classifier in classifiers:
        print("===", classifier, "===")
        y_pred = classifier.predict(X_test)

        print("Accuracy:", sklearn.metrics.accuracy_score(y_test, y_pred))
        print("Precision:", sklearn.metrics.precision_score(y_test, y_pred))
        print("Recall:", sklearn.metrics.recall_score(y_test, y_pred))
        print("F1 score:", sklearn.metrics.f1_score(y_test, y_pred))
        print("Info: Ratio of non-essential genes in the dataset:", (classes == 0).sum().iloc[0]/len(classes))
        print("\n")    

        y_proba = classifier.predict_proba(X_test)[:, 1]
        display = PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax, name=str(classifier))
        per_class_metrics(y_test, y_pred)


def per_class_metrics(y_true, y_pred):
    """
    Prints accuracy and precision for essential (1) and nonessential (0) classified genes.
    """
    for label, name in zip([1, 0], ["Essential", "Nonessential"]):
        idx = (y_pred == label)
        if idx.sum() == 0:
            print(f"No genes classified as {name}.")
            continue
        
        # Precision: Of all predicted as label, how many are truly label
        # Recall: Of all true label genes, how many were predicted as label
        prec = (y_true[idx] == label).mean()
        recall = (y_pred[y_true == label] == label).mean() if (y_true == label).sum() > 0 else 0
        
        print(f"{name} classified genes:")
        print(f"  Precision: {prec:.3f}")
        print(f"  Recall: {recall:.3f}")
        print()



def get_class_predictions_balanced(embeddings):

    dataset = embeddings.merge(classes, left_index=True, right_index=True)
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]


    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_data = pd.concat([X_train, y_train], axis=1)
    essential = train_data[train_data.iloc[:, -1] == 1]
    nonessential = train_data[train_data.iloc[:, -1] == 0]

    essential_upsampled = essential.sample(n=len(nonessential), replace=True, random_state=42)
    balanced_train = pd.concat([essential_upsampled, nonessential])
    balanced_train = balanced_train.sample(frac=1, random_state=42)

    X_train_bal = balanced_train.iloc[:, :-1]
    y_train_bal = balanced_train.iloc[:, -1]

    classifiers = [
        RandomForestClassifier(100), 
        LogisticRegression(max_iter=500)
    ]

    fig, ax = plt.subplots()

    for classifier in classifiers:
        classifier.fit(X_train_bal, y_train_bal)
        print("===", classifier, "===")
        y_pred = classifier.predict(X_test)

        print("Accuracy:", sklearn.metrics.accuracy_score(y_test, y_pred))
        print("Precision:", sklearn.metrics.precision_score(y_test, y_pred))
        print("Recall:", sklearn.metrics.recall_score(y_test, y_pred))
        print("F1 score:", sklearn.metrics.f1_score(y_test, y_pred))
        print("Info: Ratio of non-essential genes in the balanced dataset:", (y_bal == 0).sum() / len(y_bal))
        print("\n")    

        y_proba = classifier.predict_proba(X_test)[:, 1]
        PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax, name=str(classifier))
        per_class_metrics(y_test, y_pred)


def get_class_predictions_bootstrapped(embeddings, n_bootstraps=5):
    """
    Uses bootstrapping to create balanced training data and evaluate classifiers.
    """
    dataset = embeddings.merge(classes, left_index=True, right_index=True)
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]
    
    # Create a fixed test set that won't change across bootstraps
    X_train_full, X_test, y_train_full, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    classifiers = [
        RandomForestClassifier(100, random_state=42), 
        LogisticRegression(max_iter=500, random_state=42)
    ]
    
    fig, ax = plt.subplots()
    
    # For each classifier, train on multiple bootstrapped samples and average results
    for classifier in classifiers:
        print(f"=== {classifier} with Bootstrapping ===")
        
        # Store predictions across bootstraps
        all_probas = np.zeros((len(X_test), n_bootstraps))
        
        # Create and train on multiple bootstrapped balanced datasets
        for i in range(n_bootstraps):
            # Create bootstrapped balanced training data
            train_data = pd.concat([X_train_full, y_train_full], axis=1)
            essential = train_data[train_data.iloc[:, -1] == 1]
            nonessential = train_data[train_data.iloc[:, -1] == 0]
            
            # Bootstrap samples with replacement
            essential_boot = essential.sample(n=len(essential), replace=True, random_state=i+42)
            nonessential_boot = nonessential.sample(n=len(essential), replace=True, random_state=i+100)
            
            # Combine and shuffle
            balanced_boot = pd.concat([essential_boot, nonessential_boot])
            balanced_boot = balanced_boot.sample(frac=1, random_state=i)
            
            X_boot = balanced_boot.iloc[:, :-1]
            y_boot = balanced_boot.iloc[:, -1]
            
            # Train classifier on this bootstrap sample
            boot_clf = sklearn.base.clone(classifier)
            boot_clf.fit(X_boot, y_boot)
            
            # Store probability predictions
            all_probas[:, i] = boot_clf.predict_proba(X_test)[:, 1]
        
        # Average probabilities across bootstraps
        y_proba = all_probas.mean(axis=1)
        
        # Get binary predictions at threshold 0.5
        y_pred = (y_proba >= 0.5).astype(int)
        
        # Print metrics
        print("Accuracy:", sklearn.metrics.accuracy_score(y_test, y_pred))
        print("Precision:", sklearn.metrics.precision_score(y_test, y_pred))
        print("Recall:", sklearn.metrics.recall_score(y_test, y_pred))
        print("F1 score:", sklearn.metrics.f1_score(y_test, y_pred))
        print("\n")
        
        # Plot precision-recall curve
        PrecisionRecallDisplay.from_predictions(
            y_test, y_proba, ax=ax, name=f"{classifier.__class__.__name__}"
        )
        per_class_metrics(y_test, y_pred)