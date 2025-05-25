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

    data = pd.concat([X, y], axis=1)
    essential = data[data.iloc[:, -1] == 1]
    nonessential = data[data.iloc[:, -1] == 0]

    essential_upsampled = essential.sample(n=len(nonessential), replace=True, random_state=42)
    balanced_data = pd.concat([essential_upsampled, nonessential])

    balanced_data = balanced_data.sample(frac=1, random_state=42)

    X_bal = balanced_data.iloc[:, :-1]
    y_bal = balanced_data.iloc[:, -1]

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
    )

    classifiers = [
        RandomForestClassifier(100), 
        LogisticRegression(max_iter=500)
    ]

    fig, ax = plt.subplots()

    for classifier in classifiers:
        classifier.fit(X_train, y_train)
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