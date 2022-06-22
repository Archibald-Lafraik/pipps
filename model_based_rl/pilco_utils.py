import jax.lax
from jax import vmap, jit
import numpy as np
import jax.numpy as jnp

from trans_model import trans_output
from neural_nets import nn_policy


def get_trajectories(
    theta,
    betas,
    trans_noise,
    env,
    model_d1, model_d2,
    model_d3, model_d4,
    horizon,
    state_epsilons,
    trans_epsilons,
):
    start_states = jnp.array([env.reset() for _ in range(state_epsilons.shape[0])])
    time_steps = jnp.arange(horizon - 1)
    traj_func = vmap(
                get_trajectory,
                (
                    None, None, None, 0, None, None,
                    None, None, None, 0, 0
                )
            )

    trajectories = traj_func(
        theta,
        betas,
        trans_noise,
        start_states,
        model_d1, model_d2,
        model_d3, model_d4,
        time_steps,
        state_epsilons,
        trans_epsilons,
    )

    return trajectories

@jit
def get_trajectory(
    theta,
    betas,
    trans_noise,
    start_state,
    model_d1, model_d2,
    model_d3, model_d4,
    time_steps,
    state_epsilons,
    trans_epsilons,
):
    L1 = jnp.linalg.cholesky(model_d1[1])
    L2 = jnp.linalg.cholesky(model_d2[1])
    L3 = jnp.linalg.cholesky(model_d3[1])
    L4 = jnp.linalg.cholesky(model_d4[1])

    def body(prev_state, t):
        state_eps = state_epsilons[t]

        action = nn_policy(prev_state, theta)
        model_input = jnp.stack([prev_state, jnp.full((4,), action), jnp.ones((4,))]).T
        
        w_d1 = model_d1[0] + L1 @ trans_epsilons[t, 0]
        w_d2 = model_d2[0] + L2 @ trans_epsilons[t, 1]
        w_d3 = model_d3[0] + L3 @ trans_epsilons[t, 2]
        w_d4 = model_d4[0] + L4 @ trans_epsilons[t, 3]
              
        state_diff = trans_output(w_d1, w_d2, w_d3, w_d4, model_input)

        next_mean = state_diff + prev_state
        next_state = next_mean + state_eps * trans_noise

        return next_state, next_state

    _, next_states = jax.lax.scan(body, start_state, time_steps)
    states = jnp.zeros_like(state_epsilons)

    states = states.at[0].set(start_state)
    states = states.at[1:].set(next_states)

    return states


def rollout_episode(env, horizon, replay_buffer, weights):
    cur_state = env.reset()
    done = False
    
    reward = 0
    for t in range(horizon):
        prev_state = cur_state
        action = nn_policy(prev_state, weights)
        action = np.array([action])

        cur_state, _, done, _ = env.step(action)
        if not done:
            reward = t + 1
        replay_buffer.push(prev_state, action.squeeze(), cur_state)

    return reward
