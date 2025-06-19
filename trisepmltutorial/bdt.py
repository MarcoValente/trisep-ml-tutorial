import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV
import scipy.stats as stats

from trisepmltutorial.plotting import plot_train_vs_test, plot_roc_curve, plot_confusion_matrix

def train_bdt(
    X_train,
    y_train,
    output_dir='bdt',
    features=None
    ):
    # output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    xgb = XGBClassifier(
        tree_method='hist', # 'gpu_hist' for GPU acceleration
        eval_metric='logloss'
    )
    # See https://xgboost.readthedocs.io/en/stable/parameter.html for more parameters

    print('Model architecture:')
    print(xgb)

    starting_time = time.time()

    xgb.fit(X_train, y_train.values)

    training_time = time.time() - starting_time
    print(f"Training time: {training_time:.2f} seconds")

    # Save the trained model
    xgb.save_model(os.path.join(output_dir, 'model.json'))

    np.savez(
        os.path.join(output_dir, 'training_history.npz'),
        feature_names=features if features is not None else [f'feature_{i}' for i in range(X_train.shape[1])],
    )

    return xgb

def plot_learning_curve(
    X_train, y_train,
    output_dir='bdt'
    ):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    xgb = XGBClassifier(
        tree_method='hist',
        eval_metric='logloss'
    )

    train_sizes = np.linspace(0.1, 1.0, 6)
    train_scores = []
    val_scores = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for frac in train_sizes:
        n_train = int(frac * len(X_train))
        X_sub = X_train[:n_train]
        y_sub = y_train[:n_train]

        train_score = []
        val_score = []

        for train_idx, val_idx in skf.split(X_sub, y_sub):
            X_tr, X_val = X_sub[train_idx], X_sub[val_idx]
            y_tr, y_val = y_sub[train_idx], y_sub[val_idx]

            xgb.fit(X_tr, y_tr)
            train_score.append(xgb.score(X_tr, y_tr))
            val_score.append(xgb.score(X_val, y_val))

        train_scores.append(np.mean(train_score))
        val_scores.append(np.mean(val_score))

    plt.figure()
    plt.plot(train_sizes, train_scores, label='Training Score')
    plt.plot(train_sizes, val_scores, label='Validation Score')
    plt.xlabel('Training Size (fraction)')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'learning_curve.png'))

def hyperparameter_tuning(
    X_train, y_train,
    X_test, y_test,
    ):
    """
    Perform hyperparameter tuning for the BDT model.
    """
    print("Starting hyperparameter tuning...")
    # specify parameters, range, and distributions to sample from
    param_dist_XGB = {
        'max_depth': stats.randint(3, 12), # default 6
        'n_estimators': stats.randint(300, 800), #default 100
        'learning_rate': stats.uniform(0.1, 0.5), #def 0.3
    }

    gsearch = RandomizedSearchCV(
        estimator = XGBClassifier(tree_method="hist",eval_metric='logloss'),
        param_distributions = param_dist_XGB,
        scoring='roc_auc',
        n_iter=10,
        cv=2
    )

    gsearch.fit(X_train, y_train.values)

    print("Best parameters found: ", gsearch.best_params_)
    print("Best score: ", gsearch.best_score_)

    y_pred_test = gsearch.predict_proba(X_test)[:, 1].ravel()
    print("Score on test dataset: ",roc_auc_score(y_true=y_test.values, y_score=y_pred_test))
    dfsearch=pd.DataFrame.from_dict(gsearch.cv_results_)
    print(dfsearch)