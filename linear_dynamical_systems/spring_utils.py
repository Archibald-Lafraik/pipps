import numpy as np
import jax.numpy as jnp
import jax.random as jrandom

from constants import RAND_KEY

def get_A(k, delta_t, m, z):
    delta_t2 = jnp.power(delta_t, 2)
    A = jnp.array(
        [[1 - (delta_t2 * k) / m, delta_t - (delta_t2 * z) / m],
        [-delta_t * k / m, 1 - delta_t * z / m]]
    )

    return A

def get_observations(k, h, m, z, mu0, V0, trans_noise, obs_noise, num_steps, N, key=RAND_KEY):
    # Perhaps not a problem, but possibly should use same espilon samples as in get_z()
    # Just split in two batches: epsilons_trans, epsilons_obs
    zs = np.zeros((num_steps, N, 2))
    xs = np.zeros((num_steps, N, 2))

    L0 = np.linalg.cholesky(V0)
    L_trans = np.linalg.cholesky(trans_noise)
    L_obs = np.linalg.cholesky(obs_noise)

    epsilons = jrandom.normal(key=key, shape=(num_steps, N, 2))

    z0 = mu0 + epsilons[0] @ L0.T
    # x0 = z0 + epsilons[num_steps] @ L_obs.T
    x0 = z0 + epsilons[0] @ L_obs.T

    zs[0] = z0
    xs[0] = x0

    A = get_A(k, h, m, z)
    for i in range(1, num_steps):
        zs[i] = zs[i - 1] @ A.T + epsilons[i] @ L_trans.T
        # xs[i] = zs[i] + epsilons[num_steps + i] @ L_obs.T
        xs[i] = zs[i] + epsilons[i] @ L_obs.T

    return zs, xs

def get_zs(A, mu0, V0, trans_noise, num_steps, N, key=RAND_KEY):
    zs = np.zeros((num_steps, N, 2))

    L0 = np.linalg.cholesky(V0)
    L = np.linalg.cholesky(trans_noise)

    epsilons = jrandom.normal(key=key, shape=(num_steps, N, 2))

    z0 = mu0 + epsilons[0] @ L0.T
    zs[0] = z0

    for i in range(1, num_steps):
        zs[i] = zs[i - 1] @ A.T + epsilons[i] @ L.T

    return zs, epsilons