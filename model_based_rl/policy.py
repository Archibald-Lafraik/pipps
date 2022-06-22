import jax.numpy as jnp
import jax.lax
import jax.scipy
import jax.random as jrandom
from jax import vmap, jit, grad

from trans_model import trans_output
from neural_nets import nn_policy


def cost_function(state):
    target = jnp.array([0., 0., 1.])
    state_val = jnp.array([state[0], jnp.sin(state[1]), jnp.cos(state[1])])
    diff = state_val - target
    width = 1.
    return 1 - jnp.exp(- ((diff[0] ** 2 + diff[1] ** 2 + diff[2] ** 2) / width) ** 2)


def evaluate_policy(theta, beta, trans_models, env, N, horizon, key):
    epsilons = jrandom.normal(key=key, shape=(N, horizon, 4))
    model_d1, model_d2, model_d3, model_d4 = trans_models

    value_func = vmap(
        trajectory_value,
        (None, None, None, None, None, None, None, None, 0)
    )

    rewards = value_func(theta, beta, env, model_d1, model_d2, model_d3, model_d4, horizon, epsilons)        
    return rewards.mean()

@jit
def trajectory_value(
    params,
    betas, 
    trans_noise,
    start_state,
    model_d1,model_d2,
    model_d3, model_d4,
    time_steps,
    state_epsilons,
    trans_epsilons,
):
    L1 = jnp.linalg.cholesky(model_d1[1])
    L2 = jnp.linalg.cholesky(model_d2[1])
    L3 = jnp.linalg.cholesky(model_d3[1])
    L4 = jnp.linalg.cholesky(model_d4[1])


    def body(carry, t):
        prev_state, cost = carry
        state_eps = state_epsilons[t]

        action = nn_policy(prev_state, params)
        model_input = jnp.stack([prev_state, jnp.full((4,), action), jnp.ones((4,))]).T
        
        w_d1 = model_d1[0] + L1 @ trans_epsilons[t, 0]
        w_d2 = model_d2[0] + L2 @ trans_epsilons[t, 1]
        w_d3 = model_d3[0] + L3 @ trans_epsilons[t, 2]
        w_d4 = model_d4[0] + L4 @ trans_epsilons[t, 3]
        
        state_diff = trans_output(w_d1, w_d2, w_d3, w_d4, model_input)

        next_mean = state_diff + prev_state
        next_state = next_mean + state_eps * trans_noise

        cost += cost_function(next_state)

        return (next_state, cost), next_state

    init = (start_state, cost_function(start_state))
    carry, states = jax.lax.scan(body, init, time_steps)

    cost = carry[1]

    return cost


####################################### RP GRADIENTS ###################################

def policy_grad(
    params,
    betas,
    trans_noise,
    model_d1, model_d2,
    model_d3, model_d4,
    env,
    horizon,
    state_epsilons,
    trans_epsilons,
):
    start_states = jnp.array([env.reset() for _ in range(state_epsilons.shape[0])])
    time_steps = jnp.arange(horizon - 1)

    grads_func = vmap(
                grad(trajectory_value, 0),
                in_axes=(
                    None, None, None, 0, None, None,
                    None, None, None, 0, 0
                )
            )

    grads = grads_func(
        params,
        betas,
        trans_noise,
        start_states,
        model_d1, model_d2,
        model_d3, model_d4,
        time_steps,
        state_epsilons,
        trans_epsilons,
    )
    grads['mlp/~/linear_0']['w'] = grads['mlp/~/linear_0']['w'].mean(axis=0)
    grads['mlp/~/linear_0']['b'] = grads['mlp/~/linear_0']['b'].mean(axis=0)
    grads['mlp/~/linear_1']['w'] = grads['mlp/~/linear_1']['w'].mean(axis=0)
    grads['mlp/~/linear_1']['b'] = grads['mlp/~/linear_1']['b'].mean(axis=0)

    return grads
 

################################### LIKELIHOOD RATIO GRADIENTS #################################


@jit
def logpdf(
    params,
    betas,
    trans_noise,
    model_d1, model_d2,
    model_d3, model_d4,
    time_steps,
    trajectory,
    trans_epsilons,
):
    L1 = jnp.linalg.cholesky(model_d1[1])
    L2 = jnp.linalg.cholesky(model_d2[1])
    L3 = jnp.linalg.cholesky(model_d3[1])
    L4 = jnp.linalg.cholesky(model_d4[1])
    
    def body(prob, t):
        action = nn_policy(trajectory[t - 1], params)
        model_input = jnp.stack([trajectory[t - 1], jnp.full((4,), action), jnp.ones((4,))]).T
        
        w_d1 = model_d1[0] + L1 @ trans_epsilons[t, 0]
        w_d2 = model_d2[0] + L2 @ trans_epsilons[t, 1]
        w_d3 = model_d3[0] + L3 @ trans_epsilons[t, 2]
        w_d4 = model_d4[0] + L4 @ trans_epsilons[t, 3]

        state_diff = trans_output(w_d1, w_d2, w_d3, w_d4, model_input)

        next_mean = state_diff + trajectory[t - 1]

        prob += jax.scipy.stats.multivariate_normal.logpdf(
            trajectory[t],
            mean=next_mean,
            cov= jnp.eye(4) * (trans_noise ** 2),
        )

        return prob, None
    

    init = 0
    logpdf_zs, _ = jax.lax.scan(body, init, time_steps)

    return logpdf_zs


@jit
def get_sequence_rewards(trajectory):
    rewards = jnp.array([cost_function(state) for state in trajectory])
    return rewards.sum()


def lr_gradients(
    theta,
    beta,
    trans_noise,
    model_d1, model_d2,
    model_d3, model_d4,
    horizon,
    trajectories,
    trans_epsilons,
):
    time_steps = jnp.arange(1, horizon)
    grad_func = vmap(
                grad(logpdf, 0),
                (
                    None, None, None, None, None,
                    None, None, None, 0, 0
                )
            )
    lr_grads = grad_func(
        theta,
        beta,
        trans_noise,
        model_d1, model_d2,
        model_d3, model_d4,
        time_steps,
        trajectories,
        trans_epsilons,
    )
    rewards = vmap(get_sequence_rewards, (0,))(trajectories)
    lr_grads['mlp/~/linear_0']['w'] = (lr_grads['mlp/~/linear_0']['w'] * rewards[:, jnp.newaxis, jnp.newaxis]).mean(axis=0)
    lr_grads['mlp/~/linear_0']['b'] = (lr_grads['mlp/~/linear_0']['b'] * rewards[:, jnp.newaxis]).mean(axis=0)
    lr_grads['mlp/~/linear_1']['w'] = (lr_grads['mlp/~/linear_1']['w'] * rewards[:, jnp.newaxis, jnp.newaxis]).mean(axis=0)
    lr_grads['mlp/~/linear_1']['b'] = (lr_grads['mlp/~/linear_1']['b'] * rewards[:, jnp.newaxis]).mean(axis=0)

    return lr_grads
