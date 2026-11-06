import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

SEED = 42
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]


def set_seed(seed=SEED):
    keras.utils.set_random_seed(seed)


def load_data(val_size=5000):
    """Load CIFAR-10 and split into train / validation / test."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train, y_test = y_train.flatten(), y_test.flatten()
    x_val, y_val = x_train[-val_size:], y_train[-val_size:]
    x_train, y_train = x_train[:-val_size], y_train[:-val_size]
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def plot_samples(x, y, n=10, path=None):
    plt.figure(figsize=(14, 3))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(x[i])
        plt.title(CLASS_NAMES[y[i]], fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150)
    plt.show()


def merge_histories(*histories):
    merged = {}
    for h in histories:
        for k, v in h.history.items():
            merged.setdefault(k, []).extend(v)
    return merged


def plot_history(history, title, path=None):
    """history: keras History or dict."""
    h = history.history if hasattr(history, "history") else history
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(h["accuracy"], label="train")
    ax[0].plot(h["val_accuracy"], label="val")
    ax[0].set_title(f"{title} - Accuracy"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    ax[1].plot(h["loss"], label="train")
    ax[1].plot(h["val_loss"], label="val")
    ax[1].set_title(f"{title} - Loss"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150)
    plt.show()


def evaluate_model(model, x_test, y_test, title, path=None):
    """Print test metrics, classification report, and draw confusion matrix."""
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"[{title}] Test loss: {loss:.4f} | Test accuracy: {acc:.4f}")
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"Confusion Matrix - {title}")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150)
    plt.show()
    return loss, acc
