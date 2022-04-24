import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from reparametrization import get_rp_grad_var, get_rp_gradients
from scoregradient import get_lr_grad_var, get_lr_gradients
from constants import NUM_BATCHES, RAND_KEY

def get_combined_gradients(means, variances, xs, w, N, num_inputs, key):
    # Gradients (num_inputs, N)
    rp_mu_grads, rp_var_grads = get_rp_gradients(means, variances, w, N, num_inputs, key=key)
    lr_mu_grads, lr_var_grads = get_lr_gradients(means, variances, xs, w, N, num_inputs, baseline=True, key=key)

    # (2, num_inputs, N)
    rp_grads = jnp.stack([rp_mu_grads, rp_var_grads])
    lr_grads = jnp.stack([lr_mu_grads, lr_var_grads])

    # Gradient variances (num_inputs,)
    rp_mu_var, rp_var_var = get_rp_grad_var(means, variances, w, N, num_inputs)
    lr_mu_var, lr_var_var = get_lr_grad_var(means, variances, xs, w, N, num_inputs, baseline=True)

    # (2, num_inputs)
    rp_vars = jnp.stack([rp_mu_var, rp_var_var])
    lr_vars = jnp.stack([lr_mu_var, lr_var_var])

    # Compute inverse variance ratios
    inv_rp_var = jnp.reciprocal(rp_vars)
    inv_lr_var = jnp.reciprocal(lr_vars)
    inv_total_var = jnp.reciprocal(jnp.add(inv_rp_var, inv_lr_var))

    # (2, num_inputs)
    rp_ratio = jnp.multiply(inv_total_var, inv_rp_var)
    lr_ratio = 1 - rp_ratio

    # Compute combined gradients
    weighted_rp_grads = jnp.multiply(rp_ratio, rp_grads.mean(axis=1))
    weighted_lr_grads = jnp.multiply(lr_ratio, lr_grads.mean(axis=1))
    combined_grads = jnp.add(weighted_rp_grads, weighted_lr_grads)
    
    return combined_grads[0], combined_grads[1]

def get_combined_grad_var(means, variances, xs, w, N, num_inputs, num_batches=NUM_BATCHES):
    mu_grads = np.zeros((num_batches, num_inputs))
    var_grads = np.zeros((num_batches, num_inputs))

    subkeys = jrandom.split(RAND_KEY, num=num_batches)

    for i in range(num_batches):
        mu_grad, var_grad = get_combined_gradients(
                                    means,
                                    variances,
                                    xs,
                                    w,
                                    N,
                                    num_inputs,
                                    subkeys[i],
                                )
        mu_grads[i] = mu_grad
        var_grads[i] = var_grad
    
    mu_var = mu_grads.var(axis=0)
    var_var = var_grads.var(axis=0)
    
    return mu_var, var_var
