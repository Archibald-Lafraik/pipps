import jax.numpy as jnp
import numpy as np

def normalize_probs(probs):
    np_probs = np.zeros_like(probs)
    np_probs[0] = round(probs[0], 4)
    np_probs[1] = 1.0 - np_probs[0]
    return np_probs