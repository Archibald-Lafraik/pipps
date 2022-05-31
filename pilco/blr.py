import scipy.stats
import jax.numpy as jnp
import jax.random as jrandom
from jax import grad, vmap, jit

class BLR:

    def __init__(self, num_features, alpha, beta) -> None:
        self.num_features = num_features
        self.alpha = alpha
        self.beta = beta
        self.mean = jnp.zeros(num_features)
        self.cov_inv = jnp.eye(num_features) / alpha
        self.cov = jnp.eye(num_features) * alpha

    def posterior(self, phiX, y):
        cov_inv = self.cov_inv + self.beta * phiX @ phiX.T
        # TODO consider replacing by more efficient inverse method
        cov = jnp.linalg.inv(cov_inv)

        mean = cov @ (self.cov_inv @ self.mean + self.beta * phiX @ y)

        self.cov_inv = cov_inv
        self.cov = cov
        self.mean = mean

        return mean, cov

    def predict(self, phi_Xstar):
        y_pred_mean = phi_Xstar.T @ self.mean
        w_cov = self.cov
        y_pred_var = 1 / self.beta + (phi_Xstar.T @ w_cov * phi_Xstar.T).sum(axis=1)
    
        return y_pred_mean + jrandom.normal() * jnp.sqrt(y_pred_var)
