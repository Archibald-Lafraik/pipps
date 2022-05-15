from jax import vmap, jit
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom

from constants import g, RAND_KEY, m1, m2, l1, l2


def get_G(z, t):
    t1, t2, v1, v2 = z
    
    M = jnp.array([
        [(m1 + m2) * l1, m2 * l2 * jnp.cos(t1 - t2)],
        [l1 * jnp.cos(t1 - t2), l2]
    ])
    F = jnp.array([
        -m2 * l2 * (v2 ** 2) * jnp.sin(t1 - t2) - (m1 + m2) * g * jnp.sin(t1),
        l1 * (v1 ** 2) * jnp.sin(t1 - t2) - g * jnp.sin(t2)
    ])

    M_inv = jnp.linalg.inv(M)
    acc1, acc2 = M_inv @ F

    return jnp.array([v1, v2, acc1, acc2])

@jit
def rk_step(z, t, dt):
    k1 = get_G(z, t)
    k2 = get_G(z + 0.5 * k1 * dt, t + 0.5 * dt)
    k3 = get_G(z + 0.5 * k2 * dt, t + 0.5 * dt)
    k4 = get_G(z + k3 * dt, t+dt)

    return dt * (k1 + 2 * k2 + 2 * k3 + k4) /6


def get_rk_state_sequence(
    delta_t, mu0, V0, trans_noise,
    obs_noise, num_steps, 
    N, key=RAND_KEY
):
    states = np.zeros((num_steps, N, 4))
    obs = np.zeros((num_steps, N, 4))
    
    L0 = jnp.linalg.cholesky(V0)
    L_trans = jnp.linalg.cholesky(trans_noise)
    L_obs = jnp.linalg.cholesky(obs_noise)

    epsilons = jrandom.normal(key=key, shape=(num_steps, N, 4))

    z0 = np.tile(mu0, (N, 1)) + epsilons[0] @ L0.T
    x0 = z0 + epsilons[0] @ L_obs.T

    states[0] = z0
    obs[0] = x0

    z = z0
    time = jnp.linspace(delta_t, int(delta_t * (num_steps - 1)), num_steps - 1)
    for i, t in enumerate(time):
        step_func = lambda z, t, dt: rk_step(z, t, dt)
        z = states[i] + vmap(step_func, (0, None, None))(states[i], t, delta_t)

        states[i + 1] = z + epsilons[i + 1] @ L_trans.T
        obs[i + 1] = states[i + 1] + epsilons[i + 1] @ L_obs.T

    return states, obs, epsilons 


def get_coordinates(seq):
    s = seq.shape
    xs = np.zeros((s[0], s[1], 2))
    ys = np.zeros((s[0], s[1], 2))
    for i in range(s[0]):
        xs[i, :, 0] = l1 * jnp.sin(seq[i, :, 0])
        xs[i, :, 1] = xs[i, :, 0] + l2 * jnp.sin(seq[i, :, 1])

        ys[i, :, 0] = -l1 * jnp.cos(seq[i, :, 0])
        ys[i, :, 1] = ys[i, :, 0] - l2 * jnp.cos(seq[i, :, 1])

    return xs, ys
    