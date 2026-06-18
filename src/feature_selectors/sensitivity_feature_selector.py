import numpy as np

from feature_selectors.base_models.deep_selector import BaseDeepSelector


class SensitivityFeatureSelector(BaseDeepSelector):
    """
    Sensitivity analysis (Vanilla Gradients) for feature selection.

    Trains a neural network classifier and computes the mean absolute gradient
    of the predicted class score w.r.t. each input feature across all samples.
    A large gradient means a small perturbation to that feature causes a large
    change in the prediction — the feature is highly influential.

        Importance[i] = (1/N) Σ_n |∂f(x_n) / ∂x_n,i|

    The gradient is computed in a single batched GradientTape call: no
    per-sample Python loop. This works because samples in a batch share no
    activations, so ∂score_i(x_i)/∂x_j,k = 0 when i ≠ j.

    Parameters
    ----------
    n_features : int or None
        Number of features to select; None ranks all features.
    hidden_dim : int
        Size of the first hidden layer (second is hidden_dim // 2).
    epochs : int
        Maximum training epochs.
    batch_size : int
        Mini-batch size.
    random_state : int
        Seed for TF and NumPy RNGs.
    """

    def __init__(self, n_features=None, hidden_dim=128, epochs=100,
                 batch_size=16, random_state=42):
        super().__init__(n_features, random_state)
        self._hidden_dim = hidden_dim
        self._epochs = epochs
        self._batch_size = batch_size

    def fit(self, X, y):
        self.check_already_fitted()
        self._X = X
        n_samples, n_features = X.shape

        tf, keras = self._import_tf()
        X_scaled = self._scale(X)
        y_cat, n_classes = self._encode_labels(y, keras)

        model = keras.Sequential([
            keras.layers.Input(shape=(n_features,)),
            keras.layers.Dense(self._hidden_dim, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(self._hidden_dim // 2, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(n_classes, activation='softmax'),
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(
            X_scaled, y_cat,
            epochs=self._epochs,
            batch_size=self._batch_size,
            validation_split=0.2,
            callbacks=[self._early_stopping(keras)],
            verbose=0,
        )

        # One GradientTape call for the full dataset batch
        X_tensor = tf.constant(X_scaled)
        with tf.GradientTape() as tape:
            tape.watch(X_tensor)
            preds = model(X_tensor, training=False)
            predicted_classes = tf.argmax(preds, axis=1)
            indices = tf.stack(
                [tf.range(n_samples), tf.cast(predicted_classes, tf.int32)], axis=1
            )
            scores = tf.gather_nd(preds, indices)
            total = tf.reduce_sum(scores)

        grads = tape.gradient(total, X_tensor)  # (n_samples, n_features)
        importance = np.mean(np.abs(grads.numpy()), axis=0)

        self._finalize(importance, n_features)
        return self
