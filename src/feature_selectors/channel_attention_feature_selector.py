import numpy as np

from feature_selectors.base_models.deep_selector import BaseDeepSelector


class ChannelAttentionFeatureSelector(BaseDeepSelector):
    """
    Channel attention-based feature selection.

    Learns one scalar attention weight per feature — a global, sample-agnostic
    mask applied multiplicatively to the input before the classifier layers.
    The softmax normalisation forces the network to distribute a fixed
    "attention budget" across features, favouring the most discriminative ones.

    This is *channel* (per-feature) attention as in Squeeze-and-Excitation
    networks, not sample-specific self-attention. Feature importance equals
    the learned softmax weights after training.

    Architecture:
        Input → ChannelAttention (softmax weights) → Dense → BN → Dropout
              → Dense → Dropout → Softmax output

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

        class ChannelAttentionLayer(keras.layers.Layer):
            def build(self, input_shape):
                self.attn = self.add_weight(
                    shape=(input_shape[-1],),
                    initializer='glorot_uniform',
                    trainable=True,
                    name='channel_attention',
                )

            def call(self, x):
                return x * tf.nn.softmax(self.attn)

        input_layer = keras.Input(shape=(n_features,))
        attended = ChannelAttentionLayer()(input_layer)
        x = keras.layers.Dense(self._hidden_dim, activation='relu')(attended)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.Dense(self._hidden_dim // 2, activation='relu')(x)
        x = keras.layers.Dropout(0.3)(x)
        output = keras.layers.Dense(n_classes, activation='softmax')(x)
        model = keras.Model(inputs=input_layer, outputs=output)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(
            X_scaled, y_cat,
            epochs=self._epochs,
            batch_size=self._batch_size,
            validation_split=0.2,
            callbacks=[self._early_stopping(keras)],
            verbose=0,
        )

        attn_layer = next(l for l in model.layers if isinstance(l, ChannelAttentionLayer))
        importance = tf.nn.softmax(attn_layer.attn).numpy().astype(np.float64)

        self._finalize(importance, n_features)
        return self
