import jax.lax
import jax.scipy
import jax.numpy as jnp
import jax.random as jrandom
from jax import vmap, jit

from functools import partial
from constants import RAND_KEY

@jit
def sample_features(x, num_features, lengthscale, coef, omega, phi):
    omega = omega / lengthscale
    x = x[:, jnp.newaxis]
    features = coef * (2 / num_features) ** 0.5 * jnp.cos(omega @ x + phi)
    return features

@jit
def get_next_state(theta, x, num_features, lengthscale, coef, omega, phi):
    features = sample_features(x, num_features, lengthscale, coef, omega, phi)
    next_x = theta.T @ features
    return jnp.squeeze(next_x)

@jit
def log_likelihood(
    theta, X,
    num_features,
    lengthscale,
    coef,
    trans_noise,
    obs_noise,
    start_state,
    V0,
    omega,
    phi,
    epsilons,
):
    L0 = jnp.linalg.cholesky(V0)
    L_trans = jnp.linalg.cholesky(trans_noise)

    z0 = start_state + L0 @ epsilons[0]
    prob_x = jax.scipy.stats.multivariate_normal.logpdf(
        X[0], mean=z0, cov=obs_noise
    )

    def body(carry, idx):
        z, prob_xs = carry
        next_state = get_next_state(
            theta, z, num_features, lengthscale, coef, omega[idx], phi[idx]
        )
        z = next_state + L_trans @ epsilons[idx]
        prob_xs += jax.scipy.stats.multivariate_normal.logpdf(
            X[idx], mean=z, cov=obs_noise
        )
        return (z, prob_xs), z

    init = (z0, prob_x)
    indices = jnp.arange(1, X.shape[0])
    carry, _ = jax.lax.scan(body, init, indices)

    _, prob_xs = carry
    return prob_xs
    
def elbo(
    theta, X,
    num_features,
    lengthscale, 
    coef,
    trans_noise,
    obs_noise,
    start_state,
    V0,
    omegas, phis,
    epsilons,
):
    marg_likelihood_func = vmap(
        log_likelihood,
        (None, 1, None, None, None, None, None, None, None, 2, 2, 1)
    )
    marg_likelihood = marg_likelihood_func(
        theta, X, num_features, lengthscale, coef, trans_noise,
        obs_noise, start_state, V0, omegas, phis, epsilons
    )

    return marg_likelihood.mean(axis=0)

@jit
def get_z_seq(
    theta,
    num_features,
    lengthscale,
    coef,
    trans_noise,
    start_state,
    V0,
    omega, phi,
    epsilons,
):
    L0 = jnp.linalg.cholesky(V0)
    L_trans = jnp.linalg.cholesky(trans_noise)

    zs = jnp.zeros_like(epsilons)
    z = start_state + L0 @ epsilons[0]
    zs = zs.at[0].set(z)

    for i in range(1, epsilons.shape[0]):
        pred_state = get_next_state(
            theta, z, num_features, lengthscale, coef, omega[i], phi[i] 
        )
        z = pred_state + L_trans @ epsilons[i]
        zs = zs.at[i].set(z)

    return zs

def get_zs(
    theta,
    num_features,
    lengthscale,
    coef,
    trans_noise,
    start_state,
    V0,
    omegas, phis,
    epsilons
):
    zs_func = vmap(
        get_z_seq,
        in_axes=(None, None, None, None, None, None, None, 2, 2, 1),
        out_axes=1,
    )
    zs = zs_func(
        theta, num_features, lengthscale, coef, trans_noise,
        start_state, V0, omegas, phis, epsilons
    )

    return zs