import jax.scipy
import jax.numpy as jnp
import jax.random as jrandom
from jax import grad, vmap, jit

@jit
def phi_X(X, num_features, lengthscales, coefs, omega, phi):
    omega = jnp.divide(omega, lengthscales)
    # print((omega @ X.T).shape)
    features = coefs * jnp.sqrt(2 / num_features) * jnp.cos(omega @ X.T + phi)
    return features.squeeze()

@jit
def phi_X_batch(X, num_features, lengthscales, coefs, omega, phi):
    omega = jnp.divide(omega, lengthscales)
    features = coefs * jnp.sqrt(2 / num_features) * jnp.cos(omega @ X.T + phi)
    return features.squeeze()