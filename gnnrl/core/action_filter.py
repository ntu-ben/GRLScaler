import numpy as np


def damped_action(prev: np.ndarray, new: np.ndarray, damping: float = 0.7, hyst: int = 2) -> np.ndarray:
    """Apply damping and hysteresis to action vectors.

    Parameters
    ----------
    prev : np.ndarray
        Previous replica counts per service.
    new : np.ndarray
        Proposed replica counts per service.
    damping : float
        Damping factor between 0 and 1.
    hyst : int
        Maximum allowed change per step.
    """
    if prev.shape != new.shape:
        raise ValueError("prev and new must have same shape")

    diff = new - prev
    diff = np.clip(diff, -hyst, hyst)
    return prev + diff * damping
