import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


def train_evaluate_model_basic(
    model,
    df=df,
    target=target,
    n_splits=5,
    random_state=0,
):
    # Separating features and target
    X = df.drop(columns=[target])
    y = df[target]

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Lists to store metrics for each fold
    f1_scores = []
    roc_auc_scores = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []

    # Perform k-fold cross-validation
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics and append to lists
        f1_scores.append(f1_score(y_test, y_pred))
        roc_auc_scores.append(roc_auc_score(y_test, y_pred_proba))
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred))
        recall_scores.append(recall_score(y_test, y_pred))

    # Calculate the average of each metric across all folds
    avg_metrics = {
        "Mean F1 Score": round(np.mean(f1_scores), 4),
        "Mean ROC-AUC Score": round(np.mean(roc_auc_scores), 4),
        "Mean Accuracy": round(np.mean(accuracy_scores), 4),
        "Mean Precision": round(np.mean(precision_scores), 4),
        "Mean Recall": round(np.mean(recall_scores), 4),
    }

    return avg_metrics


def train_evaluate_model_SMOTE(
    model,
    df=df,
    target=target,
    n_splits=5,
    random_state=0,
):
    # Separating features and target
    X = df.drop(columns=[target])
    y = df[target]

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Lists to store metrics for each fold
    f1_scores = []
    roc_auc_scores = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []

    for fold, (train_index, test_index) in enumerate(kf.split(X, y), 1):
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]

        # Apply resampling
        sm = SMOTE()
        X_train_oversampled, y_train_oversampled = sm.fit_resample(X_train, y_train)

        # Fit model
        model.fit(X_train_oversampled, y_train_oversampled)

        # Predict on test set
        y_pred = model.predict(X_test)

        # print(f"For fold {fold}:")
        # print(f"f-score: {f1_score(y_test, y_pred)}")
        # print(f"Accuracy: {model.score(X_test, y_test)}")

        # Calculate metrics and append to lists
        f1_scores.append(f1_score(y_test, y_pred))
        roc_auc_scores.append(roc_auc_score(y_test, y_pred))
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred))
        recall_scores.append(recall_score(y_test, y_pred))

    avg_metrics = {
        "Mean F1 Score": round(np.mean(f1_scores), 4),
        "Mean ROC-AUC Score": round(np.mean(roc_auc_scores), 4),
        "Mean Accuracy": round(np.mean(accuracy_scores), 4),
        "Mean Precision": round(np.mean(precision_scores), 4),
        "Mean Recall": round(np.mean(recall_scores), 4),
    }

    return avg_metrics
