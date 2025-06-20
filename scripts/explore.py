import os
import pandas as pd

import matplotlib.pyplot as plt
from trisepmltutorial.plotting import plot_features, plot_correlations
from trisepmltutorial.dataset import load_dataset, preprocess_dataset

from pprint import pprint

######
## Load data
dataset = load_dataset("/fast_scratch_1/TRISEP_data/BeginnerTutorial/dataWW_d1.root", "tree_event")

print(f'Loaded dataset with {len(dataset)} events.')
print(f"Dataset shape: {dataset.shape}")
print(f"Number of columns in dataset: {len(dataset.columns)}")
print(f"Columns in dataset:")
pprint(f"{dataset.columns.tolist()}")
print("First 5 rows of the dataset:")
print(dataset.head())
print("Dataset description:")
print(dataset.describe())

## Now let's make some plots!

# Features to plot
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

# Target label (0 for background, 1 for signal)
target = dataset["label"]

# Make the datafram with selected features for training
reduced_dataset = pd.DataFrame(dataset, columns=features)
print(f"Training dataset shape: {reduced_dataset.shape}")

# plot features
os.makdirs("figures")

plot_features(reduced_dataset, target)
plt.savefig("figures/features.png")

plot_correlations(reduced_dataset, target)
plt.savefig("figures/correlations.png")
