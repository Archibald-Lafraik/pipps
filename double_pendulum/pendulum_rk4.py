from jax import jit, vmap
import jax.numpy as jnp 
import jax.random as jrandom

from constants import RAND_KEY, m1, m2, l1, l2, g


def G(z, t): 
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
def RK4_step(z, t, dt):
	k1 = G(z,t)
	k2 = G(z+0.5*k1*dt, t+0.5*dt)
	k3 = G(z+0.5*k2*dt, t+0.5*dt)
	k4 = G(z+k3*dt, t+dt)

	return dt * (k1 + 2*k2 + 2*k3 + k4) /6


def get_pendulum_sequence(
    start_state,
    V0,
    trans_noise,
    obs_noise,
    num_steps,
    N,
    dt,
    key=RAND_KEY
):
    time = jnp.linspace(0, num_steps * dt, num_steps)

    states = jnp.zeros((num_steps, N, 4))
    obs = jnp.zeros((num_steps, N, 4))

    L0 = jnp.linalg.cholesky(V0)
    L_trans = jnp.linalg.cholesky(trans_noise)
    L_obs = jnp.linalg.cholesky(obs_noise)

    epsilons = jrandom.normal(key=key, shape=(num_steps, N, start_state.shape[0]))
    eps_trans = epsilons
    eps_obs = epsilons

    cur_state = jnp.tile(start_state, (N, 1)) + eps_trans[0] @ L0.T
    cur_obs = cur_state + eps_obs[0] @ L_obs.T

    states = states.at[0].set(cur_state)
    obs = obs.at[0].set(cur_obs)

    for i in range(1, num_steps):
        rk4_step = vmap(
            lambda st: RK4_step(st, time[i], dt),
            (0,),
        )
        next_pos = states[i - 1] + rk4_step(cur_state)

        cur_state = next_pos + eps_trans[i] @ L_trans.T
        cur_obs = cur_state + eps_obs[i] @ L_obs.T

        states = states.at[i].set(cur_state)
        obs = obs.at[i].set(cur_obs)

    return states, obs


def get_cos_sin_states(states):
    cos_sin = lambda a: jnp.array([
        jnp.cos(a[0]),
        jnp.sin(a[0]),
        jnp.cos(a[1]),
        jnp.sin(a[1]),
    ])
    return jnp.apply_along_axis(cos_sin, 2, states)