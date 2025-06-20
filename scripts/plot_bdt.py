from trisepmltutorial.plotting import plot_train_vs_test, plot_roc_curve, plot_confusion_matrix
import numpy as np
import os

from xgboost import XGBClassifier

from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.inspection import permutation_importance
import pandas as pd
import matplotlib.pyplot as plt

##################
# Loading data and models after training
##################
print('Loading training history and model...')
output_dir='models/bdt' #Model output directory
data_path = "data/train_val_splits.pkl"

history_path = os.path.join(output_dir, "training_history.npz")
history = np.load(history_path)
feature_names = history['feature_names']

# Loading the trained model
model_path = os.path.join(output_dir, "model.json")
model = XGBClassifier()
model.load_model(model_path)
print(model)

# Loading validation data and trained model
splitdata = pd.read_pickle(data_path)
X_train = splitdata["X_train"]
y_train = splitdata["y_train"]
X_val = splitdata["X_val"]
y_val = splitdata["y_val"]

##################
# Making plots
##################

y_pred_train = model.predict_proba(X_train)[:, 1].ravel()
y_pred_test = model.predict_proba(X_val)[:, 1].ravel()

# plot train vs test
plot_train_vs_test(
    y_pred_train, y_train, 
    y_pred_test, y_val,
    bins=25, out_range=(0, 1), density=False,
    xlabel="BDT output", ylabel="Number of Events", title="Train vs Test"
)
plt.savefig(os.path.join(output_dir, 'train_vs_test.png'))

plot_train_vs_test(
    y_pred_train, y_train, 
    y_pred_test, y_val,
    bins=25, out_range=(0, 1), density=True,
    xlabel="BDT output", ylabel="A.U.", title="Train vs Test (Normalized)"
)
plt.savefig(os.path.join(output_dir, 'train_vs_test_normalized.png'))

# Compute and plot ROC curves
fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_train)
fpr_test, tpr_test, _ = roc_curve(y_val, y_pred_test)
auc_train = auc(fpr_train, tpr_train)
auc_test = auc(fpr_test, tpr_test)

plot_roc_curve(
    [fpr_train, fpr_test],
    [tpr_train, tpr_test],
    [auc_train, auc_test],
    ['Train', 'Test']
)
plt.savefig(os.path.join(output_dir, 'roc_curve.png'))

# plot confusion matrix
cm_train = confusion_matrix(y_train, (y_pred_train > 0.5).astype(int))
cm_test = confusion_matrix(y_val, (y_pred_test > 0.5).astype(int))
plot_confusion_matrix(cm_train, cm_test)
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))

"""
Plot feature importance from the trained BDT model.
"""

plt.figure()
sorted_idx = model.feature_importances_.argsort()
plt.barh(feature_names[sorted_idx], model.feature_importances_[sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Feature Importance from BDT')
plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
plt.close()

"""
Compute and plot feature permutation importance.
"""

result = permutation_importance(
    model, X_val, y_val,
    scoring='roc_auc', n_repeats=1, random_state=42, n_jobs=-1
)

sorted_idx = result.importances_mean.argsort()
plt.figure()
plt.barh(feature_names[sorted_idx], result.importances_mean[sorted_idx])
plt.xlabel('Permutation Importance')
plt.title('Permutation Importance from BDT')
plt.savefig(os.path.join(output_dir, 'permutation_importance.png'))
plt.close()
