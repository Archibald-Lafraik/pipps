import jax.numpy as jnp
from jax import vmap



def linear_policy(state, theta):
    vec = jnp.array([1, *state])
    return vec.T @ theta


def evaluate_policy(theta, trans_models, env, N, horizon):
    value_func = vmap(trajectory_value, (None, None, None, None, None))
    value = value_func(theta, trans_models, env, horizon)
    return value.mean()


def trajectory_value(theta, trans_models, env, horizon):
    model_d1, model_d2, model_d3, model_d4 = trans_models
    cur_state = env.reset()

    for t in range(horizon):
        action = linear_policy(cur_state, theta)

        # Make model prediction return a sample?
        next_state_d1 = model_d1.predict(jnp.array([cur_state[0], action]))
        next_state_d2 = model_d2.predict(jnp.array([cur_state[1], action]))
        next_state_d3 = model_d3.predict(jnp.array([cur_state[2], action]))
        next_state_d4 = model_d4.predict(jnp.array([cur_state[3], action]))

        cur_state = jnp.array([
            next_state_d1, 
            next_state_d2,
            next_state_d3,
            next_state_d4
        ])

        # Vertical angle between pole and cart exceeds 0.2 rads
        if cur_state[1] > 0.2:
            break
    
    return t