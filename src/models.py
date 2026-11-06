from tensorflow import keras
from tensorflow.keras import layers


def build_cnn(input_shape=(32, 32, 3), num_classes=10):
    """Model 1: CNN dibangun dari nol."""
    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomTranslation(0.1, 0.1)(x)

    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.25)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="cnn_from_scratch")
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def build_transfer_model(num_classes=10, image_size=96, lr=1e-3):
    """Model 2: Transfer Learning dengan MobileNetV2 (pretrained ImageNet)."""
    base = keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # Phase 1: feature extraction

    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Resizing(image_size, image_size)(inputs)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)  # skala ke [-1, 1] sesuai MobileNetV2
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="mobilenetv2_transfer")
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model, base


def unfreeze_for_finetuning(model, base, n_last_layers=30, lr=1e-5):
    """Phase 2: fine-tuning n layer terakhir dengan learning rate kecil."""
    base.trainable = True
    for layer in base.layers[:-n_last_layers]:
        layer.trainable = False
    # BatchNorm tetap dibekukan agar statistik pretrained tidak rusak
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model
