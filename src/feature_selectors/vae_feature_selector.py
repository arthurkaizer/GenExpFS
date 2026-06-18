import numpy as np

from feature_selectors.base_models.deep_selector import BaseDeepSelector


class VAEFeatureSelector(BaseDeepSelector):
    """
    Variational Autoencoder (VAE) based feature selection.

    Extends AutoencoderFeatureSelector with a stochastic latent space.
    The encoder outputs a Gaussian distribution (μ, σ²); the latent vector
    is sampled via the reparametrisation trick. The KL divergence term
    regularises the encoder and prevents memorisation — critical for gene
    expression data with many features and few samples.

    Feature importance is the L1 norm of the first encoder layer's weights,
    using the same rationale as AutoencoderFeatureSelector.

    Loss = MSE(reconstruction) + β · KL(N(μ, σ²) ∥ N(0, 1))

    Parameters
    ----------
    n_features : int or None
        Number of features to select; None ranks all features.
    latent_dim : int
        Dimensionality of the latent space.
    hidden_dim : int
        Size of the intermediate encoder/decoder layers.
    kl_weight : float
        β — weight of the KL divergence term relative to reconstruction loss.
    epochs : int
        Maximum training epochs.
    batch_size : int
        Mini-batch size.
    random_state : int
        Seed for TF and NumPy RNGs.
    """

    def __init__(self, n_features=None, latent_dim=16, hidden_dim=128,
                 kl_weight=0.01, epochs=100, batch_size=16, random_state=42):
        super().__init__(n_features, random_state)
        self._latent_dim = latent_dim
        self._hidden_dim = hidden_dim
        self._kl_weight = kl_weight
        self._epochs = epochs
        self._batch_size = batch_size

    def fit(self, X, y=None):
        self.check_already_fitted()
        self._X = X
        n_samples, n_features = X.shape

        tf, keras = self._import_tf()
        X_scaled = self._scale(X)

        class SamplingLayer(keras.layers.Layer):
            def call(self, inputs):
                z_mean, z_log_var = inputs
                eps = tf.random.normal(tf.shape(z_mean))
                return z_mean + tf.exp(0.5 * z_log_var) * eps

        # Encoder
        enc_in = keras.Input(shape=(n_features,), name='enc_input')
        h = keras.layers.Dense(
            self._hidden_dim, activation='relu', name='enc_hidden',
            kernel_regularizer=keras.regularizers.l2(1e-4),
        )(enc_in)
        h = keras.layers.Dropout(0.3)(h)
        z_mean = keras.layers.Dense(self._latent_dim, name='z_mean')(h)
        z_log_var = keras.layers.Dense(self._latent_dim, name='z_log_var')(h)
        z = SamplingLayer()([z_mean, z_log_var])
        encoder = keras.Model(enc_in, [z_mean, z_log_var, z], name='encoder')

        # Decoder
        dec_in = keras.Input(shape=(self._latent_dim,), name='dec_input')
        h_dec = keras.layers.Dense(self._hidden_dim, activation='relu')(dec_in)
        h_dec = keras.layers.Dropout(0.3)(h_dec)
        dec_out = keras.layers.Dense(n_features, activation='linear', name='dec_output')(h_dec)
        decoder = keras.Model(dec_in, dec_out, name='decoder')

        # VAE loss must be computed inside a layer so that KerasTensors are
        # resolved to real tensors before TF ops are applied (Keras 3 / TF 2.16+).
        kl_weight = self._kl_weight

        class VAELossLayer(keras.layers.Layer):
            def call(self, inputs):
                original, reconstructed, z_mean, z_log_var = inputs
                recon = tf.reduce_mean(
                    tf.reduce_sum(tf.square(original - reconstructed), axis=1)
                )
                kl = -0.5 * tf.reduce_mean(
                    1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
                )
                self.add_loss(recon + kl_weight * kl)
                return reconstructed

        vae_in = keras.Input(shape=(n_features,))
        zm, zlv, z_sample = encoder(vae_in)
        vae_raw = decoder(z_sample)
        vae_out = VAELossLayer()([vae_in, vae_raw, zm, zlv])
        vae = keras.Model(vae_in, vae_out, name='vae')
        vae.compile(optimizer='adam')

        vae.fit(
            X_scaled,
            epochs=self._epochs,
            batch_size=self._batch_size,
            validation_split=0.2,
            callbacks=[self._early_stopping(keras)],
            verbose=0,
        )

        # Encoder first-layer kernel: shape (n_features, hidden_dim)
        enc_weights = encoder.get_layer('enc_hidden').get_weights()[0]
        importance = np.sum(np.abs(enc_weights), axis=1)

        self._finalize(importance, n_features)
        return self
