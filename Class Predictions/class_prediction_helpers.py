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

        y_proba = classifier.predict_proba(X_test)[:, 1]
        display = PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax, name=str(classifier))

