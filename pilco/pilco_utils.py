import jax.lax
from jax import vmap, jit
import numpy as np
import jax.numpy as jnp

from trans_model import predict
from policy import linear_policy
from rff import phi_X_batch


def get_trajectories(
    theta,
    beta,
    env,
    model_d1, model_d2,
    model_d3, model_d4,
    horizon,
    num_features,
    lengthscales,
    coefs,
    noise,
    state_epsilons,
    trans_epsilons,
    omegas,
    phis,
):
    # start_states = jnp.array([env.reset() for _ in range(state_epsilons.shape[0])])
    start_states = jnp.zeros((state_epsilons.shape[0], 4))
    time_steps = jnp.arange(horizon - 1)
    traj_func = vmap(
            vmap(
                get_trajectory,
                (
                    None, None, 0, None, None, None, None, None,
                    None, None, None, None, 0, None, 0, 0
                )
            ),
            in_axes=(
                None, None, None, None, None, None, None, 
                None, None, None, None, None, None, 0, None, None
            ) 
    )

    trajectories = traj_func(
        theta,
        beta,
        start_states,
        model_d1, model_d2,
        model_d3, model_d4,
        time_steps,
        num_features,
        lengthscales,
        coefs,
        noise,
        state_epsilons,
        trans_epsilons,
        omegas,
        phis,
    )

    return trajectories

@jit
def get_trajectory(
    theta,
    beta,
    start_state,
    model_d1, model_d2,
    model_d3, model_d4,
    time_steps,
    num_features,
    lengthscales,
    coefs,
    noise,
    state_epsilons,
    trans_epsilons,
    omegas,
    phis,
):

    def body(prev_state, t):
        state_eps = state_epsilons[t]
        trans_eps = trans_epsilons[t]
        omega = omegas[t]
        phi = phis[t]

        action = linear_policy(prev_state, theta)
        model_input = jnp.stack([prev_state, jnp.full((4,), action)]).T

        input = phi_X_batch(model_input, num_features, lengthscales, coefs, omega, phi)
    
        means = jnp.concatenate([model_d1[0], model_d2[0], model_d3[0], model_d4[0]])
        covs = jnp.diag(jnp.concatenate([
            jnp.diag(model_d1[1]),
            jnp.diag(model_d2[1]),
            jnp.diag(model_d3[1]),
            jnp.diag(model_d4[1])
        ]))
        d1, d2, d3, d4 = predict(means, covs, beta, input.reshape(-1), trans_eps)

        next_mean = jnp.array([d1, d2, d3, d4]) + prev_state
        next_state = next_mean + state_eps * noise

        return next_state, next_state

    states = jnp.zeros_like(state_epsilons)
    states = states.at[0].set(start_state)
    states = states.at[1:].set(jax.lax.scan(body, start_state, time_steps)[1])

    return states


def rollout_episode(env, horizon, replay_buffer, theta):
    cur_state = env.reset()
    done = False
    
    reward = 0
    for t in range(horizon):
        prev_state = cur_state
        action = linear_policy(prev_state, theta)
        action = np.array([action])

        cur_state, _, done, _ = env.step(action)
        if not done:
            reward = t + 1
        replay_buffer.push(prev_state, action.squeeze(), cur_state)

    return reward
