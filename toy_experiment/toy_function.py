import jax.numpy as jnp
import jax.random as jrandom

from constants import RAND_KEY

def foo(inputs, w):
    return 0.1 * jnp.sin(inputs * w / jnp.pi) + jnp.power((inputs/10), 2) + 0.1

def mc_function(sample_var: int, xs, w: int, N: int, num_inputs: int):
    # Compute smooth version of foo, as expectation of foo(x) with x sampled
    # normal
    means = jnp.array(xs, dtype=jnp.float32)
    cov = jnp.eye(num_inputs, num_inputs, dtype=jnp.float32) * sample_var

    mc_samples = jrandom.multivariate_normal(
                        key=RAND_KEY,
                        mean=means,
                        cov=cov,
                        shape=(N,)
    )

    return foo(mc_samples, w).mean(axis=0, keepdims=False)