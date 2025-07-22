# utils/norm.py
import numpy as np
from sklearn.preprocessing import StandardScaler

def normalize(X: np.ndarray, mode: str = "z") -> np.ndarray:
    """
    mode: z: z - center
    standard: StandardScaler
    none: nothing scaled
    """
    X = X.astype(np.float32, copy=True)

    if mode == "none":
        return X
    elif mode == "z":
        X -= X.mean(0, keepdims=True)
    else:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    return X
