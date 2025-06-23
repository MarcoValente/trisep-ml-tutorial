from trisepmltutorial.plotting import plot_training_history, plot_train_vs_test, plot_roc_curve
import numpy as np
import os
import torch

from sklearn.metrics import roc_curve, auc, roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt

##################
# Loading data and models after training
##################
print('Loading training history and model...')
output_dir='models/mlp' #Model output directory
data_path = "data/train_val_splits.pkl"

history_path = os.path.join(output_dir, "training_history.npz")
history = np.load(history_path)
training_loss_history = history['training_loss_history']
validation_loss_history = history['validation_loss_history']
num_epochs = history['num_epochs']
feature_names = history['feature_names']

# Loading the trained model
model_path = os.path.join(output_dir, "model.pth")
model = torch.load(model_path, weights_only=False)
model.eval()  # Set the model to evaluation mode
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

# Plot training and validation loss
print(f'Plotting the training and validation loss from \'{output_dir}\'')
plot_training_history(training_loss_history, validation_loss_history, num_epochs)
plt.savefig(os.path.join(output_dir, 'training_validation_loss.png'))

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)

# Move model to GPU if available
device = next(model.parameters()).device
if device.type == 'cuda':
    model.to('cuda')
    X_train_tensor_gpu = X_train_tensor.to('cuda')
    X_val_tensor_gpu = X_val_tensor.to('cuda')
    y_pred_train = model(X_train_tensor_gpu).detach().cpu().numpy().ravel()
    y_pred_test = model(X_val_tensor_gpu).detach().cpu().numpy().ravel()
else:
    y_pred_train = model(X_train_tensor).detach().numpy().ravel()
    y_pred_test = model(X_val_tensor).detach().numpy().ravel()

# plot train vs test
plot_train_vs_test(
    y_pred_train, y_train_tensor.numpy(), 
    y_pred_test, y_val_tensor.numpy(),
    bins=25, out_range=(0, 1), density=False,
    xlabel="NN output", ylabel="Number of Events", title="Train vs Test"
)
plt.savefig(os.path.join(output_dir, 'train_vs_test.png'))

plot_train_vs_test(
    y_pred_train, y_train_tensor.numpy(), 
    y_pred_test, y_val_tensor.numpy(),
    bins=25, out_range=(0, 1), density=True,
    xlabel="NN output", ylabel="A.U.", title="Train vs Test"
)
plt.savefig(os.path.join(output_dir, 'train_vs_test_normalized.png'))

# Compute and plot ROC curves

fpr_train, tpr_train, _ = roc_curve(y_train_tensor.numpy(), y_pred_train)
fpr_test, tpr_test, _ = roc_curve(y_val_tensor.numpy(), y_pred_test)
auc_train = auc(fpr_train, tpr_train)
auc_test = auc(fpr_test, tpr_test)

plot_roc_curve(
    [fpr_train, fpr_test], 
    [tpr_train, tpr_test], 
    [auc_train, auc_test], 
    labels=['Train', 'Test']
)
plt.savefig(os.path.join(output_dir, 'roc_curve.png'))

# Plot the feature importance
print('Plotting the permutation importance...')

model.eval()
device = next(model.parameters()).device

X_orig = X_val.copy()

# Baseline score
X_tensor = torch.tensor(X_orig, dtype=torch.float32).to(device)
with torch.no_grad():
    y_pred = model(X_tensor).detach().cpu().numpy().ravel()
baseline = roc_auc_score(y_val.values, y_pred)

importances = []
print(feature_names)
for i, vname in enumerate(feature_names):
    X_permuted = X_orig.copy()
    X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
    X_tensor_perm = torch.tensor(X_permuted, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_pred_perm = model(X_tensor_perm).detach().cpu().numpy().ravel()
    score = roc_auc_score(y_val.values, y_pred_perm)
    print(f'Feature: {vname}, Importance: {baseline - score}, Baseline: {baseline}, Score: {score}')
    importances.append(baseline - score)

importances = np.array(importances)
sorted_idx = np.argsort(importances)

print(f"Feature importances: {importances}")

plt.figure()
plt.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
plt.xlabel('Permutation Importance')
plt.title('Permutation Importance from MLP')
plt.savefig(os.path.join(output_dir, 'permutation_importance.png'))
plt.close()

# Feature importance using gradients

X_tensor = torch.tensor(X_val.copy(), dtype=torch.float32).to(device)
X_tensor.requires_grad = True  # Enable gradient computation
output = model(X_tensor)
output.mean().backward()  # Backprop on the mean output

# Get average absolute gradient across all samples
feature_importance = X_tensor.grad.abs().mean(dim=0).detach().to('cpu').numpy()

sorted_idx = np.argsort(feature_importance)

plt.figure()
plt.barh(np.array(feature_names)[sorted_idx], feature_importance[sorted_idx])
plt.xlabel("Feature name")
plt.ylabel("Gradient-based Importance")
plt.title("Feature Importance from Gradients")
plt.savefig(os.path.join(output_dir, "feature_importance.png"))
plt.close()
