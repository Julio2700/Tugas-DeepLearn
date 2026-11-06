"""Jalankan: python train.py --cnn-epochs 30 --tl-epochs 8 --ft-epochs 8"""
import argparse
import json
import os

from tensorflow import keras

from src.common import (set_seed, load_data, plot_samples, plot_history,
                        evaluate_model, merge_histories)
from src.models import build_cnn, build_transfer_model, unfreeze_for_finetuning


def main(args):
    set_seed()
    os.makedirs("results", exist_ok=True)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()
    print("Train:", x_train.shape, "Val:", x_val.shape, "Test:", x_test.shape)
    plot_samples(x_train, y_train, path="results/sample_images.png")

    callbacks = lambda: [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2),
    ]

    # ---------- Model 1: CNN ----------
    cnn = build_cnn()
    cnn.summary()
    h_cnn = cnn.fit(x_train, y_train, validation_data=(x_val, y_val),
                    epochs=args.cnn_epochs, batch_size=args.batch_size,
                    callbacks=callbacks())
    plot_history(h_cnn, "CNN", "results/cnn_history.png")
    _, cnn_acc = evaluate_model(cnn, x_test, y_test, "CNN", "results/cnn_confusion_matrix.png")
    cnn.save("results/cnn_model.keras")

    # ---------- Model 2: Transfer Learning ----------
    tl, base = build_transfer_model()
    h1 = tl.fit(x_train, y_train, validation_data=(x_val, y_val),
                epochs=args.tl_epochs, batch_size=args.batch_size, callbacks=callbacks())
    unfreeze_for_finetuning(tl, base)
    h2 = tl.fit(x_train, y_train, validation_data=(x_val, y_val),
                epochs=args.ft_epochs, batch_size=args.batch_size, callbacks=callbacks())
    plot_history(merge_histories(h1, h2), "MobileNetV2", "results/transfer_history.png")
    _, tl_acc = evaluate_model(tl, x_test, y_test, "MobileNetV2",
                               "results/transfer_confusion_matrix.png")
    tl.save("results/transfer_model.keras")

    with open("results/summary.json", "w") as f:
        json.dump({"cnn_test_accuracy": cnn_acc, "transfer_test_accuracy": tl_acc}, f, indent=2)
    print(f"\nCNN: {cnn_acc:.4f} | Transfer Learning: {tl_acc:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cnn-epochs", type=int, default=30)
    p.add_argument("--tl-epochs", type=int, default=8)
    p.add_argument("--ft-epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    main(p.parse_args())
