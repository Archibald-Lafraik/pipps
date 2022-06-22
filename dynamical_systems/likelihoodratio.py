import jax.scipy
import jax.lax
import jax.numpy as jnp
from jax import vmap, grad, jit

def get_likelihood(C, E, zs, xs):
    likelihood = 1
    for i in range(zs.shape[0]):
        likelihood *= jax.scipy.stats.norm.pdf(xs[i], loc= C * zs[i], scale=jnp.sqrt(E))
    return likelihood

@jit
def logpdf(mu0, V0, A, B, zs):
    joint_logpdf = jax.scipy.stats.norm.logpdf(zs[0], loc=mu0, scale=jnp.sqrt(V0))
    for i in range(1, zs.shape[0]):
        joint_logpdf += jax.scipy.stats.norm.logpdf(zs[i], loc=A * zs[i - 1], scale=jnp.sqrt(B))
    return joint_logpdf

def get_lr_gradients(mu0, V0, A, B, C, E, xs, zs):
    z_grad_func = vmap(grad(logpdf, argnums=2), in_axes=(None, None, None, None, 1))
    z_grads = z_grad_func(mu0, V0, A, B, zs)

    likelihood_func = vmap(get_likelihood, in_axes=(None, None, 1, 1))
    likelihood = likelihood_func(C, E, zs, xs)

    lr_grad = jnp.multiply(z_grads, likelihood)
    return lr_grad.mean()