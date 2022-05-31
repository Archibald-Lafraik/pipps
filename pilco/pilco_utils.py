import jax.random as jrandom
import jax.numpy as jnp

from rff import phi_X

def train_transition_models(key, replay_buffer, model_d1, model_d2, model_d3, model_d4):
    num_features = model_d1.num_features
    omega = jrandom.normal(key=key, shape=(num_features, 2))
    phi = jrandom.uniform(key=key, minval=0, maxval=2 * jnp.pi, shape=(num_features, 1))
    lengthscales = jnp.full((num_features, 1), 1.)
    coefs = jnp.full_like(lengthscales, 1.)

    X, y = replay_buffer.get_train_test_arrays()

    X_d1 = X[:, :2]
    X_d2 = X[:, 2:4]
    X_d3 = X[:, 4:6]
    X_d4 = X[:, 6:8]
    y_d1 = y[:, 0]
    y_d2 = y[:, 1]
    y_d3 = y[:, 2]
    y_d4 = y[:, 3]

    phi_X_d1 = phi_X(X_d1, num_features, lengthscales, coefs, omega, phi)
    phi_X_d2 = phi_X(X_d2, num_features, lengthscales, coefs, omega, phi)
    phi_X_d3 = phi_X(X_d3, num_features, lengthscales, coefs, omega, phi)
    phi_X_d4 = phi_X(X_d4, num_features, lengthscales, coefs, omega, phi)

    model_d1.posterior(phi_X_d1, y_d1)
    model_d2.posterior(phi_X_d2, y_d2)
    model_d3.posterior(phi_X_d3, y_d3)
    model_d4.posterior(phi_X_d4, y_d4)
