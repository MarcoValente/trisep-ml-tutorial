import matplotlib.pyplot as plt
import numpy as np

def plot_features(features_df, target):
    plt.figure()
    # background distributions
    ax = features_df[target==0].hist(
        figsize=(15,12),
        color='blue',
        alpha=0.5, 
        density=True,
        label='Background'
    )

    ax = ax.flatten()[:features_df.shape[1]]

    # signal distributions
    features_df[target==1].hist(
        figsize=(15,12),
        color='red',
        alpha=0.5,
        density=True,
        ax=ax,
        label='Signal'
    )

    # add legends
    for a in ax:
        a.legend()

def plot_correlations(features_df, target):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    corr_sig = features_df[target==1].corr()
    im0 = axes[0].imshow(corr_sig, cmap='coolwarm', vmin=-1, vmax=1)
    axes[0].set_title('Signal Correlation Matrix')
    axes[0].set_xticks(range(len(corr_sig.columns)))
    axes[0].set_xticklabels(corr_sig.columns, rotation=90)
    axes[0].set_yticks(range(len(corr_sig.columns)))
    axes[0].set_yticklabels(corr_sig.columns)
    fig.colorbar(im0, ax=axes[0])

    corr_bkg = features_df[target==0].corr()
    im1 = axes[1].imshow(corr_bkg, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1].set_title('Background Correlation Matrix')
    axes[1].set_xticks(range(len(corr_bkg.columns)))
    axes[1].set_xticklabels(corr_bkg.columns, rotation=90)
    axes[1].set_yticks(range(len(corr_bkg.columns)))
    axes[1].set_yticklabels(corr_bkg.columns)
    fig.colorbar(im1, ax=axes[1])

def plot_training_history(training_loss, validation_loss, num_epochs):
    plt.figure()
    plt.plot(range(1, num_epochs + 1), training_loss, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), validation_loss, label='Validation Loss')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()

def plot_train_vs_test(
    y_pred_train, y_train, y_pred_test, y_test,
    bins=25, out_range=(0,1), density=False,
    xlabel="", ylabel="a.u.", title=""
    ):
    plt.figure()

    plt.hist(y_pred_train[y_train == 1], bins=bins, color='r', alpha=0.5, range=out_range, histtype='stepfilled', density=density, label='S (train)')

    plt.hist(y_pred_train[y_train == 0], bins=bins, color='b', alpha=0.5, range=out_range, histtype='stepfilled', density=density, label='B (train)')

    hist, bins = np.histogram(y_pred_test[y_test==1], bins=bins, range=out_range, density=density)
    scale = len(y_pred_test[y_test==1]) / hist.sum()
    err = np.sqrt(hist*scale) / scale
    center = (bins[1:] + bins[:-1]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o', color='r', label='S (test)')

    hist, bins = np.histogram(y_pred_test[y_test==0], bins=bins, range=out_range, density=density)
    scale = len(y_pred_test[y_test==0]) / hist.sum()
    err = np.sqrt(hist*scale) / scale
    center = (bins[1:] + bins[:-1]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o', color='b', label='B (test)')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

def plot_roc_curve(fpr_arr, tpr_arr, auc_arr, labels):
    plt.figure()
    for fpr, tpr, auc, label in zip(fpr_arr, tpr_arr, auc_arr, labels):
        plt.plot(fpr, tpr, label=f'{label} (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

def plot_confusion_matrix(cm_train, cm_test):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cm_train, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Train)')
    plt.colorbar()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.xticks([0, 1], ['Background', 'Signal'])
    plt.yticks([0, 1], ['Background', 'Signal'])
    plt.subplot(1, 2, 2)
    plt.imshow(cm_test, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Test)')
    plt.colorbar()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.xticks([0, 1], ['Background', 'Signal'])
    plt.yticks([0, 1], ['Background', 'Signal'])
    plt.tight_layout()