import os
import time
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import roc_curve, auc, roc_auc_score

from trisepmltutorial.plotting import plot_training_history, plot_train_vs_test, plot_roc_curve

def train_mlp(
    X_train, X_val,
    y_train, y_val,
    # w_train, w_val,
    num_epochs=20,
    output_dir='mlp'
    ):
    # output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    # w_train_tensor = torch.tensor(w_train.values, dtype=torch.float32)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
    # w_val_tensor = torch.tensor(w_val.values, dtype=torch.float32)

    # Create DataLoader for training data
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=128)

    print('-'*100)
    print('Shape of training data:', X_train.shape)
    print('Shape of training labels:', y_train.shape)
    print('Shape of validation data:', X_val_tensor.shape)
    print('Shape of validation labels:', y_val_tensor.shape)
    print('-'*100)

    # Define a simple neural network model
    nodes_in_hidden_layers = 5
    model = nn.Sequential(
        nn.Linear(X_train.shape[1], nodes_in_hidden_layers),  # 1st hidden layer
        nn.ReLU(),  # Activation function
        nn.Linear(nodes_in_hidden_layers, nodes_in_hidden_layers),  # 2nd hidden layer
        nn.ReLU(),  # Activation function
        nn.Linear(nodes_in_hidden_layers, 1),  # Output layer
        nn.Sigmoid(),  # Sigmoid activation for binary classification
    )

    print('Model architecture:')
    print(model)

    # Binary Cross-Entropy Loss
    loss_fn = nn.BCELoss(reduction='none') # compute loss per sample

    # Using Adam optimizer
    lr=0.001
    print(f'Using Adam optimizer with learning rate {lr}')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    # Train on GPU if available
    if torch.cuda.is_available():
        # Move model and data to GPU if available
        print("Using GPU for training")
        model.to('cuda')
        X_train_tensor = X_train_tensor.to('cuda')
        y_train_tensor = y_train_tensor.to('cuda')
        X_val_tensor = X_val_tensor.to('cuda')
        y_val_tensor = y_val_tensor.to('cuda')

    device = next(model.parameters()).device

    training_loss_history = []
    validation_loss_history = []

    starting_time = time.time()

    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        # Iterate over batches
        for X_batch, y_batch in train_loader:
            # Move data to the appropriate device
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            outputs = model(X_batch).squeeze()
            loss = loss_fn(outputs, y_batch.float()).mean()
            # weighted_loss = (loss * w_batch).mean()  # Apply weights to the loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss

        train_loss = train_loss.detach().item() / len(train_loader)  # Average loss over batches

        # Validation step
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor).squeeze()
            val_loss = loss_fn(val_outputs, y_val_tensor.float()).mean()
            val_loss = val_loss.detach().item()  # Convert to scalar

        training_loss_history.append(train_loss)
        validation_loss_history.append(val_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.8f}, Validation Loss: {val_loss:.8f}")

    training_time = time.time() - starting_time
    print(f"Training time: {training_time} seconds")

    # Save the trained model
    torch.save(model, os.path.join(output_dir, 'model.pth'))
    # Save training history to a numpy file
    np.savez(
        os.path.join(output_dir, 'training_history.npz'),
        training_loss_history=np.array(training_loss_history),
        validation_loss_history=np.array(validation_loss_history),
        num_epochs=num_epochs
    )

    return model

def feature_importance(model, feature_names, X, y, output_dir='mlp'):
    # evaluate feature importance using permutation importance

    model.eval()
    device = next(model.parameters()).device

    X_orig = X.copy()

    # Baseline score
    X_tensor = torch.tensor(X_orig, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_pred = model(X_tensor).detach().cpu().numpy().ravel()
    baseline = roc_auc_score(y.values, y_pred)

    importances = []
    for i, vname in enumerate(feature_names):
        X_permuted = X_orig.copy()
        X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
        X_tensor_perm = torch.tensor(X_permuted, dtype=torch.float32).to(device)
        with torch.no_grad():
            y_pred_perm = model(X_tensor_perm).detach().cpu().numpy().ravel()
        score = roc_auc_score(y.values, y_pred_perm)
        importances.append(baseline - score)

    importances = np.array(importances)
    sorted_idx = np.argsort(importances)

    # Plot feature importance
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure()
    plt.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
    plt.xlabel('Permutation Importance')
    plt.title('Permutation Importance from MLP')
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.close()
