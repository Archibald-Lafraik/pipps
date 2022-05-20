import jax.lax
import jax.scipy
import jax.numpy as jnp
import jax.random as jrandom
from jax import vmap, jit, jacobian

from rff import likelihood, get_next_state

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
        jacobian(likelihood, 0),
        (None, 1, None, None, None, None, None, None, None, 1, 1, 1)
    )
    grads = grads_func(
        theta, X, num_features, lengthscale, coef, trans_noise,
        obs_noise, start_state, V0, omegas, phis, epsilons
    )
    return grads.mean(axis=0)


############################ Likelihood-ratio Gradients ##########################


@jit
def lr_likelihood(X, zs, obs_noise):
    init = 1

    def body(likelihood, idx):
        likelihood *= jax.scipy.stats.multivariate_normal.pdf(
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
        logpdf += jax.scipy.stats.multivariate_normal.logpdf(
            zs[idx],
            mean=get_next_state(
                theta, zs[idx - 1], num_features, lengthscale, coef, omega, phi
            ),
            cov=trans_noise
        )
        return logpdf, None

    indices = jnp.arange(1, zs.shape[0])
    logpdf, _ = jax.lax.scan(body, init, indices)

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
    zs
):
    grad_func = vmap(
        jacobian(logpdf, 0),
        in_axes=(None, None, None, None, None, None, 1, 1, None, 1)
    )
    grads = grad_func(
        theta, num_features, lengthscale, coef, start_state,
        V0, omegas, phis, trans_noise, zs)

    likelihood = vmap(lr_likelihood, in_axes=(1, 1, None))(X, zs, obs_noise)

    lr_grads = likelihood[:, jnp.newaxis, jnp.newaxis] * grads

    return lr_grads.mean(axis=0)