import numpy as np
import jax.numpy as jnp
import jax.random as jrandom

from constants import RAND_KEY

def get_samples(num_inputs, N, mu_0, V_0, A, B, C, E, key=RAND_KEY):
    samples = jrandom.normal(key=key, shape=(num_inputs, N))
    zns = np.zeros((num_inputs, N))
    xns = np.zeros((num_inputs, N))

    z0 = mu_0 + samples[0, :] * np.sqrt(V_0)
    zns[0] = z0
    xns[0] = C * z0 + samples[0, :] * np.sqrt(E)

    for i in range(1, num_inputs):
        z = A * zns[i - 1] + samples[i, :] * np.sqrt(B)
        x = C * z + samples[i, :] * np.sqrt(E)

        zns[i] = z
        xns[i] = x

    return jnp.asarray(zns), jnp.asarray(xns)

def get_z_samples(num_inputs, N, mu_0, V_0, A, B, key=RAND_KEY):
    samples = jrandom.normal(key=key, shape=(num_inputs, N))
    zs = np.zeros((num_inputs, N))

    z0 = mu_0 + samples[0, :] * np.sqrt(V_0)
    zs[0] = z0
    for i in range(1, num_inputs):
        zs[i] = A * zs[i - 1] + samples[i] * np.sqrt(B)
    
    return zs, samples 
