"""
solution3_neural/cnn_keras.py
-----------------------------
OPTIONAL deep-learning benchmark: a small Convolutional Neural Network that
learns its own features directly from the pixels (no handcrafted features).

Written with Keras 3, which runs on top of TensorFlow, PyTorch or JAX:
    pip install keras tensorflow          # or: keras torch   /  keras "jax[cpu]"
Choose the backend with the KERAS_BACKEND environment variable
(default: tensorflow).

Architecture (about 30k parameters, trains on a laptop CPU in minutes)
    input 96x96x3
    random flip / rotation / contrast       (data augmentation, training only)
    [Conv 3x3 16 -> BatchNorm -> ReLU -> MaxPool]
    [Conv 3x3 32 -> BatchNorm -> ReLU -> MaxPool]
    [Conv 3x3 64 -> BatchNorm -> ReLU -> MaxPool]
    GlobalAveragePooling -> Dropout 0.3 -> Dense softmax

The convolution layers play the role of feature extraction and the last
dense layer is the classifier, both trained together by backpropagation.
"""

import numpy as np

import config


def keras_available():
    try:
        import keras  # noqa: F401
        return True
    except Exception:
        return False


def resize_batch(images, size):
    """Resize uint8 images to (size, size) and scale to [0, 1] floats."""
    from PIL import Image
    if images.shape[1] == size:
        return images.astype("float32") / 255.0
    return np.stack([np.asarray(Image.fromarray(im).resize((size, size), Image.BILINEAR))
                     for im in images]).astype("float32") / 255.0


def build_cnn(n_classes, size=96):
    import keras
    from keras import layers

    def block(x, filters):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization(momentum=0.9)(x)   # faster-adapting running stats (small datasets)
        x = layers.Activation("relu")(x)
        return layers.MaxPooling2D()(x)

    inp = keras.Input((size, size, 3))
    x = layers.RandomFlip("horizontal_and_vertical")(inp)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomContrast(0.15)(x)
    for f in (16, 32, 64):
        x = block(x, f)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inp, out, name="leaf_cnn")
    model.compile(optimizer=keras.optimizers.Adam(config.CNN_LR),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


class CNNClassifier:
    """Thin wrapper giving the CNN the same fit / predict / predict_proba API."""

    def __init__(self, n_classes, size=96, epochs=config.CNN_EPOCHS,
                 batch_size=config.CNN_BATCH, seed=config.RANDOM_STATE):
        self.n_classes, self.size = n_classes, size
        self.epochs, self.batch_size, self.seed = epochs, batch_size, seed

    def fit(self, images, y):
        import keras
        keras.utils.set_random_seed(self.seed)
        self.model = build_cnn(self.n_classes, self.size)
        X = resize_batch(images, self.size)
        stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                             restore_best_weights=True)
        h = self.model.fit(X, y, validation_split=0.1, epochs=self.epochs,
                           batch_size=self.batch_size, callbacks=[stop], verbose=2)
        self.history_ = {"train_loss": h.history["loss"], "val_loss": h.history["val_loss"],
                         "train_acc": h.history["accuracy"], "val_acc": h.history["val_accuracy"]}
        return self

    def predict_proba(self, images):
        return self.model.predict(resize_batch(images, self.size), verbose=0)

    def predict(self, images):
        return np.argmax(self.predict_proba(images), axis=1)

    def n_params(self):
        return int(self.model.count_params())
