import jax.lax
from jax import vmap, jit
import numpy as np
import jax.random as jrandom
import jax.numpy as jnp

from trans_model import predict
from policy import linear_policy, nonlinear_policy
from rff import phi_X_batch
from rbf import rbf_policy
from neural_nets import nn_policy


def get_trajectories(
    theta,
    betas,
    env,
    model_d1, model_d2,
    model_d3, model_d4,
    horizon,
    noise,
    state_epsilons,
    trans_epsilons,
):
    # start_states = jnp.array([env.reset() for _ in range(state_epsilons.shape[0])])
    start_states = jnp.zeros((state_epsilons.shape[0], 4))
    time_steps = jnp.arange(horizon - 1)
    traj_func = vmap(
                get_trajectory,
                (
                    None, None, 0, None, None, None,
                    None, None, None, 0, 0
                )
            )

    trajectories = traj_func(
        theta,
        betas,
        start_states,
        model_d1, model_d2,
        model_d3, model_d4,
        time_steps,
        noise,
        state_epsilons,
        trans_epsilons,
    )

    return trajectories

@jit
def get_trajectory(
    theta,
    betas,
    start_state,
    model_d1, model_d2,
    model_d3, model_d4,
    time_steps,
    noise,
    state_epsilons,
    trans_epsilons,
):

    def body(prev_state, t):
        state_eps = state_epsilons[t]
        trans_eps = trans_epsilons[t]

        action = nonlinear_policy(prev_state, theta)
        model_input = jnp.stack([prev_state, jnp.full((4,), action), jnp.ones((4,))]).T
        
        d1 = predict(*model_d1, betas[0], model_input[0], trans_eps[0])
        d2 = predict(*model_d2, betas[1], model_input[1], trans_eps[1])
        d3 = predict(*model_d3, betas[2], model_input[2], trans_eps[2])
        d4 = predict(*model_d4, betas[3], model_input[3], trans_eps[3])

        next_mean = jnp.array([d1, d2, d3, d4]) + prev_state
        next_state = next_mean + state_eps * noise

        return next_state, next_state

    states = jnp.zeros_like(state_epsilons)
    states = states.at[0].set(start_state)
    states = states.at[1:].set(jax.lax.scan(body, start_state, time_steps)[1])

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
