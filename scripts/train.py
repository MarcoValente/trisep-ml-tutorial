import pandas as pd

import matplotlib.pyplot as plt
from trisepmltutorial.plotting import plot_features, plot_correlations
from trisepmltutorial.dataset import load_dataset, preprocess_dataset
import pickle
import os

######
## Load data
dataset = load_dataset("data/dataWW_d1.root", "tree_event")

######
# Features to train on
features = [
    "met_et",
    "met_phi",
    "lep_pt_0",
    "lep_pt_1",
    "lep_eta_0",
    "lep_eta_1",
    "lep_phi_0",
    "lep_phi_1",
    "jet_n",
    "jet_pt_0",
    "jet_pt_1",
]

# Target label
target = dataset["label"]

# Make the datafram with selected features for training
dataset_train = pd.DataFrame(dataset, columns=features)
print(f"Training dataset shape: {dataset_train.shape}")
print('First 5 rows of the training dataset:')
print(dataset_train.head())

######
# Preprocess data
# split dataset into training and test sets
test_size = 0.25  # 25% of the data will be used for testing and validation
X_train, X_val, X_test, y_train, y_val, y_test = preprocess_dataset(dataset_train, target, test_size=test_size)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Save train/val splits to file for later use

split_data = {
    "X_train": X_train,
    "X_val": X_val,
    "y_train": y_train,
    "y_val": y_val,
}
train_val_datafname = "data/train_val_splits.pkl"
if not os.path.exists(train_val_datafname):
    with open(train_val_datafname, "wb") as f:
        pickle.dump(split_data, f)
    print(f"Saved train/val splits to {train_val_datafname}")

######
# Train model
###

###
# Neural network
do_nn = True
if do_nn:
    import trisepmltutorial.mlp as mynn
    outdir_mlp = 'models/mlp'

    # Train the neural network
    mlp = mynn.train_mlp(X_train, X_val, y_train, y_val, output_dir=outdir_mlp)

# Boosted decision trees
do_bdt = False
if do_bdt:
    import trisepmltutorial.bdt as mybdt

    outdir_bdt = "models/bdt"

    # Train the BDT model
    bdt = mybdt.train_bdt(X_train, y_train, output_dir=outdir_bdt)

    # Evaluate the BDT model
    mybdt.evaluate_bdt(bdt, X_train, X_test, y_train, y_test, output_dir=outdir_bdt)

    # Feature importance
    mybdt.feature_importance(bdt, dataset_train.columns, output_dir=outdir_bdt)
    mybdt.feature_permutation(
        bdt, dataset_train.columns, X_test, y_test, output_dir=outdir_bdt
    )

    # Learning curve
    mybdt.plot_learning_curve(X_train, y_train, output_dir=outdir_bdt)

    # Hyperparameter tuning
    mybdt.hyperparameter_tuning(X_train, y_train, X_test, y_test)

    # %%% Exercise %%%
    # Hyperparameter tuning:
    # - Try different values for `max_depth`, `n_estimators`, `learning_rate`
    # - Use `GridSearchCV` or `RandomizedSearchCV` from `sklearn` to find the best parameters

    # %%% Exercise %%%
    # Try other BDT implementations:
    # - `SKLearn`'s `GBDT`
    # - `LightGBM`

# %%% Exercise %%%
# Compare the performance of different models:
# - Compare the AUC scores of BDT and MLP models
# - Compare the training time of BDT and MLP models
