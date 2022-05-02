import jax.scipy
import jax.lax
import jax.numpy as jnp
import numpy as np
from jax import vmap, grad
import scipy.stats


def get_likelihood_ratio_gradient(A, B, C, E, zs, xs, num_inputs, N, baseline=False): 
    log_pdf = lambda z, A, z_prev, B: jax.scipy.stats.norm.logpdf(z, loc=A * z_prev, scale=jnp.sqrt(B))

    A_grad_logpdf = vmap(
        vmap(fun=grad(log_pdf, 1), in_axes=(0, None, 0, None)),
        in_axes=(0, None, 0, None)
    )
    
    marg_likelihood = scipy.stats.norm.pdf(xs, loc=C * zs, scale=np.sqrt(E))

    if baseline:
        marg_likelihood = marg_likelihood - marg_likelihood.mean(axis=1)

    A_grads = np.zeros((num_inputs, N))
    A_grads[1:] = A_grad_logpdf(zs[1:], A, zs[:-1], B)
    lr_gradient = jnp.multiply(A_grads, marg_likelihood)

    return lr_gradient.mean(axis=1)
