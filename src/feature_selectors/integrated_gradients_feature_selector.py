import numpy as np

from feature_selectors.base_models.deep_selector import BaseDeepSelector


class IntegratedGradientsFeatureSelector(BaseDeepSelector):
    """
    Integrated Gradients (Sundararajan et al., 2017) for feature selection.

    Vanilla gradients saturate when the network is confident: the gradient
    approaches zero near a flat region even if the feature was decisive in
    getting there. IG fixes this by integrating the gradient along the
    straight-line path from a zero baseline to the actual input.

        IG[i] = (x_i − 0) × ∫₀¹ ∂f(α·x)/∂x_i dα

    This satisfies the completeness axiom: Σ_i IG[i] equals the difference
    in model output between the input and the baseline.

    Implementation: all (n_steps × n_samples) interpolated inputs are passed
    to the model in a single forward pass and one GradientTape call.

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
    n_steps : int
        Number of integration steps (Riemann approximation).
    random_state : int
        Seed for TF and NumPy RNGs.
    """

    def __init__(self, n_features=None, hidden_dim=128, epochs=100,
                 batch_size=16, n_steps=50, random_state=42):
        super().__init__(n_features, random_state)
        self._hidden_dim = hidden_dim
        self._epochs = epochs
        self._batch_size = batch_size
        self._n_steps = n_steps

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

        X_tf = tf.constant(X_scaled)  # (n_samples, n_features)

        # Fix the target class per sample using the original (α=1) input,
        # so all interpolated versions of sample i share the same target.
        original_preds = model(X_tf, training=False)
        predicted_classes = tf.argmax(original_preds, axis=1)  # (n_samples,)

        # Build all interpolated inputs in one tensor:
        # α ∈ [0, 1] → shape (n_steps, n_samples, n_features) → (n_steps*n_samples, n_features)
        alphas = tf.reshape(tf.linspace(0.0, 1.0, self._n_steps), (-1, 1, 1))
        all_inputs = tf.reshape(alphas * tf.expand_dims(X_tf, 0), (-1, n_features))

        # Tile the target classes to match the interpolated batch
        tiled_classes = tf.tile(predicted_classes, [self._n_steps])  # (n_steps*n_samples,)

        with tf.GradientTape() as tape:
            tape.watch(all_inputs)
            preds = model(all_inputs, training=False)
            total_rows = tf.shape(all_inputs)[0]
            indices = tf.stack(
                [tf.range(total_rows), tf.cast(tiled_classes, tf.int32)], axis=1
            )
            scores = tf.gather_nd(preds, indices)
            total = tf.reduce_sum(scores)

        all_grads = tape.gradient(total, all_inputs)                        # (n_steps*n, n_features)
        all_grads = tf.reshape(all_grads, (self._n_steps, n_samples, n_features))
        mean_grads = tf.reduce_mean(all_grads, axis=0)                      # (n_samples, n_features)

        ig = mean_grads * X_tf                                               # × (x − baseline=0)
        importance = tf.reduce_mean(tf.abs(ig), axis=0).numpy()

        self._finalize(importance, n_features)
        return self
