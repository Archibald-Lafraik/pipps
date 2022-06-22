from tracemalloc import start
import jax.lax
import jax.scipy
import jax.numpy as jnp
import jax.random as jrandom
from jax import vmap, jit, jacobian

from rff import log_likelihood, get_next_state

############################ Reparameterized Gradients ################################

def rp_gradients(
    theta, 
    X, 
    num_features, 
    lengthscale,
    coef,
    trans_noise,
    obs_noise,
    start_state,
    V0,
    omegas,
    phis,
    epsilons,
):
    grads_func = vmap(
        jacobian(log_likelihood, 0),
        (None, 1, None, None, None, None, None, None, None, 2, 2, 1)
    )
    # print(f'RP: {likelihood(theta, X[:, 0], num_features, lengthscale, coef, trans_noise, obs_noise, start_state, V0, omegas[:, 0], phis[:, 0], epsilons[:, 0])}')
    grads = grads_func(
        theta, X, num_features, lengthscale, coef, trans_noise,
        obs_noise, start_state, V0, omegas, phis, epsilons
    )
    return grads.mean(axis=0)


############################ Likelihood-ratio Gradients ##########################


@jit
def lr_log_likelihood(X, zs, obs_noise):
    init = 0

    def body(likelihood, idx):
        likelihood += jax.scipy.stats.multivariate_normal.logpdf(
            X[idx], mean=zs[idx], cov=obs_noise
        )
        return likelihood, None

    indices = jnp.arange(zs.shape[0])
    likelihood, _ = jax.lax.scan(body, init, indices)

    return likelihood

@jit
def logpdf(
    theta,
    num_features,
    lengthscale,
    coef,
    start_state,
    V0,
    omega,
    phi,
    trans_noise,
    zs
):
    init = jax.scipy.stats.multivariate_normal.logpdf(
        zs[0], mean=start_state, cov=V0
    )
    
    def body(logpdf, idx):
        predicted_state = get_next_state(
                theta, zs[idx - 1], num_features, lengthscale, coef, omega[idx], phi[idx]
            )
        logpdf += jax.scipy.stats.multivariate_normal.logpdf(
            zs[idx],
            mean=predicted_state,
            cov=trans_noise
        )
        return logpdf, None

    indices = jnp.arange(1, zs.shape[0])
    logpdf, _ = jax.lax.scan(body, init, indices)

    return logpdf

@jit
def epsilons_logpdf(
    theta,
    num_features,
    lengthscale,
    coef,
    start_state,
    V0,
    omega,
    phi,
    trans_noise,
    # epsilons
    zs
):
    L0 = jnp.linalg.cholesky(V0)
    L_trans = jnp.linalg.cholesky(trans_noise)

    # zs = jnp.zeros_like(epsilons)
    # z = start_state + L0 @ epsilons[0]
    # zs = zs.at[0].set(z)

    logpdf = jax.scipy.stats.multivariate_normal.logpdf(
        # z, mean=start_state, cov=V0
        zs[0], mean=start_state, cov=V0
    )

    # for i in range(1, epsilons.shape[0]):
    for i in range(1, zs.shape[0]):
        pred_state = get_next_state(theta, zs[i-1], num_features, lengthscale, coef, omega[i], phi[i])
        # z = pred_state + L_trans @ epsilons[i]
        # zs = zs.at[i].set(z)
        logpdf += jax.scipy.stats.multivariate_normal.logpdf(
            zs[i],
            mean=pred_state, cov=trans_noise
        )

    return logpdf


def lr_gradients(
    theta, 
    X, 
    num_features, 
    lengthscale,
    coef,
    trans_noise,
    obs_noise,
    start_state,
    V0,
    omegas,
    phis,
    zs,
    epsilons
):
    grad_func = vmap(
        jacobian(logpdf, 0),
        # logpdf,
        in_axes=(None, None, None, None, None, None, 2, 2, None, 1)
    )
    grads = grad_func(
        theta, num_features, lengthscale, coef, start_state,
        V0, omegas, phis, trans_noise, zs
    )

    likelihood = vmap(lr_log_likelihood, in_axes=(1, 1, None))(X, zs, obs_noise)
    
    lr_grads = likelihood[:, jnp.newaxis, jnp.newaxis] * grads
  
    return lr_grads.mean(axis=0)