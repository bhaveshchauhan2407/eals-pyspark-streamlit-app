import numpy as np


def predict_score(user_vector: np.ndarray, item_vector: np.ndarray) -> float:
    """
    Dot product between user and item latent vectors.
    Equivalent to the Java predict(u, i) logic.
    """
    return float(np.dot(user_vector, item_vector))