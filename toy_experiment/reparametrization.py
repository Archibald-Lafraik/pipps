import jax
from jax import grad
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom

from constants import RAND_KEY
from constants import NUM_BATCHES
from toy_function import foo


def rp_foo(mean, var, w, eps):
    rp_sample = mean + jnp.sqrt(var) * eps
    return foo(rp_sample, w)


def get_rp_gradients(means, variances, w, N, num_inputs, key=RAND_KEY):
    epsilons = jrandom.normal(key=key, shape=(num_inputs,N))
    # epsilons = jnp.tile(epsilons, (num_inputs, 1))

    mu_grad = jax.vmap(
        jax.vmap(grad(rp_foo, 0), in_axes=(0, 0, None, 0)), in_axes=(0, 0, None, 0)
    )
    var_grad = jax.vmap(
        jax.vmap(grad(rp_foo, 1), in_axes=(0, 0, None, 0)), in_axes=(0, 0, None, 0)
    )

    mu_grads = mu_grad(means, variances, w, epsilons)
    var_grads = var_grad(means, variances, w, epsilons)

    return mu_grads, var_grads

def get_rp_grad_var(means, variances, w, N, num_inputs, num_batches=NUM_BATCHES):
    mu_grads = np.zeros((num_batches, num_inputs))
    var_grads = np.zeros((num_batches, num_inputs))

    subkeys = jrandom.split(RAND_KEY, num=num_batches)

    for i in range(num_batches):
        mu_grad, var_grad = get_rp_gradients(
                                    means,
                                    variances,
                                    w,
                                    N,
                                    num_inputs,
                                    subkeys[i],
                                )
        mu_grads[i] = mu_grad.mean(axis=1)
        var_grads[i] = var_grad.mean(axis=1)
    
    mu_var = mu_grads.var(axis=0)
    var_var = var_grads.var(axis=0)
    
    return mu_var, var_var