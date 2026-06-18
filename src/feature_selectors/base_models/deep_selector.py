import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

from .base_selector import BaseSelector, ResultType


class BaseDeepSelector(BaseSelector):
    """
    Base class for deep learning-based feature selectors.

    Provides shared helpers used by all subclasses:
      - lazy TensorFlow import (only fails at fit-time, not import-time)
      - deterministic seeding
      - input scaling and label encoding
      - early stopping callback
      - final weight/rank/support assignment
    """

    result_type = ResultType.WEIGHTS

    def __init__(self, n_features=None, random_state=42):
        super().__init__(n_features)
        self._random_state = random_state

    def _import_tf(self):
        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError:
            raise ImportError("TensorFlow is required: pip install 'tensorflow>=2.10'")
        tf.random.set_seed(self._random_state)
        np.random.seed(self._random_state)
        return tf, keras

    def _scale(self, X):
        scaler = StandardScaler()
        return scaler.fit_transform(X).astype(np.float32)

    def _encode_labels(self, y, keras):
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        n_classes = len(np.unique(y_enc))
        y_cat = keras.utils.to_categorical(y_enc, num_classes=n_classes)
        return y_cat, n_classes

    def _early_stopping(self, keras, patience=15):
        return keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience, restore_best_weights=True
        )

    def _finalize(self, weights, n_features):
        self._weights = np.asarray(weights, dtype=np.float64)
        self._rank = np.argsort(self._weights)[::-1]
        if self._n_features is not None:
            self._selected = self._rank[:self._n_features]
            self._support_mask = np.zeros(n_features, dtype=bool)
            self._support_mask[self._selected] = True
        self._fitted = True
