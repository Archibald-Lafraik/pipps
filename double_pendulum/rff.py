from tracemalloc import start
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
    features = coef * (2 / num_features) ** 0.5 * jnp.cos(omega @ x.T + jnp.tile(phi, (1, x.shape[0])))
    return features

@jit
def get_next_state(theta, x, num_features, lengthscale, coef, omega, phi):
    features = sample_features(x, num_features, lengthscale, coef, omega, phi)
    next_x = features.T @ theta
    return next_x 

@jit
def likelihood(
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

    z0 = start_state + jnp.sqrt(L0) @ epsilons[0]
    prob_x = jax.scipy.stats.multivariate_normal.pdf(X[0], mean=z0, cov=obs_noise)

    def body(carry, idx):
        z, prob_xs = carry
        next_z = get_next_state(
            theta, z, num_features, lengthscale, coef, omega, phi 
        )
        z = next_z + L_trans @ epsilons[idx]
        prob_xs *= jax.scipy.stats.multivariate_normal.pdf(
            X[idx], mean=z, cov=obs_noise
        )
        return (z, prob_xs), None

    init = (z0, prob_x)
    indices = jnp.arange(1, X.shape[0])
    carry, _ = jax.lax.scan(body, init, indices)

    _, prob_xs = carry
    return prob_xs
    
def marginal_likelihood(
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
        likelihood,
        (None, 1, None, None, None, None, None, None, None, 1, 1, 1)
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

    z0 = start_state + epsilons[0] @ jnp.sqrt(L0)

    def body(z, idx):
        next_z = get_next_state(
            theta, z, num_features, lengthscale, coef, omega, phi
        )
        z = next_z + epsilons[idx] @ L_trans
        return z, z

    init = z0
    indices = jnp.arange(1, epsilons.shape[0])
    _, zs = jax.lax.scan(body, init, indices)

    return zs, epsilons

def get_zs(
    num_steps, N,
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
        in_axes=(None, None, None, None, None, None, None, 1, 1, 1),
        out_axes=1,
    )
    zs = zs_func(
        theta, num_features, lengthscale, coef, trans_noise,
        start_state, V0, omegas, phis, epsilons
    )

    return zs