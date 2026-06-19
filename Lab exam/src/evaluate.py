import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report
)


def evaluate_model(model, X_train, X_test, y_train, y_test, label='Model'):
    """Train model, measure time, and return all evaluation metrics."""
    
    # Train
    train_start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - train_start

    # Predict
    infer_start = time.time()
    y_pred = model.predict(X_test)
    infer_time = time.time() - infer_start

    # Metrics
    accuracy   = accuracy_score(y_test, y_pred)
    f1_macro   = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"\n=== {label} ===")
    print(f"Accuracy   : {accuracy:.4f}")
    print(f"F1 Macro   : {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")
    print(f"Train Time : {train_time:.4f}s")
    print(classification_report(y_test, y_pred, zero_division=0))

    return {
        'label': label,
        'model': model,
        'y_pred': y_pred,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'train_time': train_time,
        'infer_time': infer_time
    }


def plot_confusion_matrix(y_test, y_pred, class_names, title, save_path=None):
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig