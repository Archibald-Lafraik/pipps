import jax.scipy
import jax.numpy as jnp
from jax import jacobian, vmap, jit

from polynomial_model import logpdf_z

@jit
def lr_log_likelihood(
    obs_noise,
    xs,
    zs,
):
    def body(prob_xs, idx):
        prob_xs += jax.scipy.stats.multivariate_normal.logpdf(
            xs[idx], mean=zs[idx], cov=obs_noise
        )
        return prob_xs, None

    init = 0
    indices = jnp.arange(zs.shape[0])
    prob_xs, _ = jax.lax.scan(body, init, indices)

    return prob_xs



def lr_gradients(
    theta,
    start_state,
    V0,
    trans_noise,
    obs_noise,
    xs,
    zs,
):
    grads_func = vmap(
        jacobian(logpdf_z, 0),
        (None, None, None, None, 1)
    )
    grads = grads_func(
        theta, start_state, V0, trans_noise, zs
    )
    log_likelihood = vmap(lr_log_likelihood, (None, 1, 1))(obs_noise, xs, zs)

    lr_grads = log_likelihood[:, jnp.newaxis] * grads

    return lr_grads.mean(axis=0)

