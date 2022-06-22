import jax.numpy as jnp
from jax import jit, vmap

@jit
def get_basis(x, u, sigma):
    return jnp.exp(-((x - u) / (2 * sigma)) ** 2)

@jit
def multidim_basis(x, u, sigma):
    cov = sigma.reshape((x.shape[0], x.shape[0]))
    cov_inv = jnp.linalg.inv(cov)
    return jnp.exp(-0.5 * (x - u).T @ cov_inv @ (x - u))

@jit
def rbf_out(x, theta):
    bases_func = vmap(get_basis, (None, 0, 0))
    bases = bases_func(x, theta[:, 1], theta[:, 2]) * theta[:, 0]
    return jnp.sum(bases)

@jit
def rbf_policy(state, theta):
    bases_func = vmap(multidim_basis, (None, 0, 0))
    bases = theta[:, 0] * bases_func(state, theta[:, 1], theta[:, 2:])
    return jnp.clip(jnp.sum(bases), -1., 1.)


def get_loss(xs, ys, theta):
    l = 0
    for idx, x in enumerate(xs):
        x = xs[idx]
        pred = rbf_out(x, theta)
        l += (ys[idx] - pred) ** 2

    return l