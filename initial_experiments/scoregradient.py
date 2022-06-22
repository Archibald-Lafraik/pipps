import jax
import jax.scipy
import jax.numpy as jnp
import jax.random as jrandom
from jax import grad
import numpy as np
from constants import NUM_BATCHES

from toy_function import foo, linear_foo
from constants import RAND_KEY


def get_lr_gradients(
    means,
    variances,
    xs,
    w,
    N,
    num_inputs,
    baseline=False,
    key=RAND_KEY,
):
    variance = variances[0][0]

    samples = jrandom.multivariate_normal(
        key=key,
        mean=jnp.array(xs, dtype=jnp.float32),
        cov=jnp.eye(num_inputs, num_inputs, dtype=jnp.float32) * variance,
        shape=(N,),
    ).T

    log_pdf = jax.scipy.stats.norm.logpdf

    mu_grad_logpdf = jax.vmap(
        jax.vmap(grad(log_pdf, 1), in_axes=(0, 0, 0)), in_axes=(0, 0, 0)
    )
    var_grad_logpdf = jax.vmap(
        jax.vmap(grad(log_pdf, 2), in_axes=(0, 0, 0)), in_axes=(0, 0, 0)
    )

    f = foo(samples, w)

    if baseline:
        f = f - f.mean(axis=1)

    stds = jnp.sqrt(variances)
    mu_grads = jnp.multiply(mu_grad_logpdf(samples, means, stds), f)
    var_grads = jnp.multiply(var_grad_logpdf(samples, means, stds), f)

    return mu_grads, var_grads

def get_lr_grad_var(
    means,
    variances,
    xs,
    w,
    N,
    num_inputs,
    num_batches=NUM_BATCHES,
    baseline=False,
):
    mu_grads = np.zeros((num_batches, num_inputs))
    var_grads = np.zeros((num_batches, num_inputs))

    subkeys = jrandom.split(RAND_KEY, num=num_batches)

    for i in range(num_batches):
        mu_grad, var_grad = get_lr_gradients(
                                    means,
                                    variances,
                                    xs,
                                    w,
                                    N,
                                    num_inputs,
                                    baseline,
                                    subkeys[i],
                                )
        mu_grads[i] = mu_grad.mean(axis=1)
        var_grads[i] = var_grad.mean(axis=1)
    
    mu_var = mu_grads.var(axis=0)
    var_var = var_grads.var(axis=0)
    
    return mu_var, var_var


# Used for testing LR gradient estimator values

def get_lr_linear_foo_gradients(
    means,
    variances,
    xs,
    N,
    num_inputs,
    baseline=False,
    key=RAND_KEY,
):
    variance = variances[0][0]
    stds = jnp.sqrt(variances)
        
    # epsilons = jnp.tile(jrandom.normal(key=key, shape=(N,)), (num_inputs, 1))
    # epsilons = jrandom.normal(key=key, shape=(num_inputs, N))
    # samples = jnp.multiply(stds, epsilons) + means

    samples = jrandom.multivariate_normal(
        key=key,
        mean=jnp.array(means[:,0], dtype=jnp.float32),
        cov=jnp.eye(num_inputs, num_inputs, dtype=jnp.float32) * variance,
        shape=(N,),
    ).T

    log_pdf = jax.scipy.stats.norm.logpdf

    mu_grad_logpdf = jax.vmap(
        jax.vmap(grad(log_pdf, 1), in_axes=(0, 0, 0)), in_axes=(0, 0, 0)
    )
    var_grad_logpdf = jax.vmap(
        jax.vmap(grad(log_pdf, 2), in_axes=(0, 0, 0)), in_axes=(0, 0, 0)
    )

    f = linear_foo(samples)

    if baseline:
        f = f - f.mean(axis=1)

    mu_grads = jnp.multiply(mu_grad_logpdf(samples, means, stds), f)
    var_grads = jnp.multiply(var_grad_logpdf(samples, means, stds), f)

    return mu_grads, var_grads

def get_lr_linear_foo_grad_var(
    means,
    variances,
    xs,
    N,
    num_inputs,
    num_batches=NUM_BATCHES,
    baseline=False,
):
    mu_grads = np.zeros((num_batches, num_inputs))
    var_grads = np.zeros((num_batches, num_inputs))

    subkeys = jrandom.split(RAND_KEY, num=num_batches)

    for i in range(num_batches):
        mu_grad, var_grad = get_lr_linear_foo_gradients(
                                    means,
                                    variances,
                                    xs,
                                    N,
                                    num_inputs,
                                    baseline,
                                    subkeys[i],
                                )
        mu_grads[i] = mu_grad.mean(axis=1)
        var_grads[i] = var_grad.mean(axis=1)
    
    mu_var = mu_grads.var(axis=0)
    var_var = var_grads.var(axis=0)
    
    return mu_var, var_var