import numpy as np

from feature_selectors.base_models.deep_selector import BaseDeepSelector


class AutoencoderFeatureSelector(BaseDeepSelector):
    """
    Unsupervised autoencoder-based feature selection.

    Trains a denoising autoencoder to compress and reconstruct the input.
    Feature importance is the L1 norm of the first encoder layer's weights:
    features with larger absolute weights contribute more to building the
    latent representation and are therefore considered more informative.

    Architecture: Input → Dense(hidden_dim) → Dropout → Dense(bottleneck_dim)
                         → Dense(hidden_dim) → Dropout → Dense(n_features)

    Parameters
    ----------
    n_features : int or None
        Number of features to select; None ranks all features.
    bottleneck_dim : int
        Size of the compressed latent representation.
    hidden_dim : int
        Size of the intermediate encoder/decoder layers.
    epochs : int
        Maximum training epochs (early stopping may stop earlier).
    batch_size : int
        Mini-batch size.
    random_state : int
        Seed for TF and NumPy RNGs.
    """

    def __init__(self, n_features=None, bottleneck_dim=32, hidden_dim=128,
                 epochs=100, batch_size=16, random_state=42):
        super().__init__(n_features, random_state)
        self._bottleneck_dim = bottleneck_dim
        self._hidden_dim = hidden_dim
        self._epochs = epochs
        self._batch_size = batch_size

    def fit(self, X, y=None):
        self.check_already_fitted()
        self._X = X
        n_samples, n_features = X.shape

        tf, keras = self._import_tf()
        X_scaled = self._scale(X)

        model = keras.Sequential([
            keras.layers.Input(shape=(n_features,)),
            keras.layers.Dense(
                self._hidden_dim, activation='relu', name='enc_hidden',
                kernel_regularizer=keras.regularizers.l2(1e-4),
            ),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(self._bottleneck_dim, activation='relu', name='bottleneck'),
            keras.layers.Dense(self._hidden_dim, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(n_features, activation='linear'),
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(
            X_scaled, X_scaled,
            epochs=self._epochs,
            batch_size=self._batch_size,
            validation_split=0.2,
            callbacks=[self._early_stopping(keras)],
            verbose=0,
        )

        # Encoder first-layer kernel: shape (n_features, hidden_dim)
        # L1 norm over output neurons → importance per input feature
        enc_weights = model.get_layer('enc_hidden').get_weights()[0]
        importance = np.sum(np.abs(enc_weights), axis=1)

        self._finalize(importance, n_features)
        return self
