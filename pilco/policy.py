import jax.numpy as jnp
import numpy as np
import jax.lax
import jax.scipy
import jax.random as jrandom
from jax import vmap, jit, jacobian

from trans_model import predict
from rff import phi_X_batch


def cost_function(state):
    target = jnp.array([0., 0., 1.])
    state_val = jnp.array([state[0], jnp.sin(state[1]), jnp.cos(state[1])])
    diff = state_val - target
    width = 1.
    return 1 - jnp.exp(- ((diff[0] ** 2 + diff[1] ** 2 + diff[2] ** 2) / width) ** 2)


def linear_policy(state, theta):
    vec = jnp.array([1, *state])
    return jnp.clip(vec.T @ theta, -1., 1.)


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
    theta,
    beta, 
    start_state,
    model_d1,model_d2,
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

    def body(carry, t):
        prev_state, cost = carry
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

        cost += cost_function(next_state)

        return (next_state, cost), next_state

    init = (start_state, cost_function(start_state))
    carry, states = jax.lax.scan(body, init, time_steps)
    cost = carry[1]

    return cost

####################################### RP GRADIENTS ###################################

def policy_grad(
    theta,
    beta,
    model_d1, model_d2,
    model_d3, model_d4,
    env,
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

    grads_func = vmap(
            vmap(
                jacobian(trajectory_value, 0),
                # trajectory_value,
                in_axes=(
                    None, None, 0, None, None, None, None, None,
                    None, None, None, None, 0, None, 0, 0
                )
            ),
            in_axes=(
                None, None, None, None, None, None, None, None,
                None, None, None, None, None, 0, None, None,
            )
    )

    grads = grads_func(
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

    return grads.mean(axis=(1, 0)) 
 

################################### LIKELIHOOD RATIO GRADIENTS #################################


@jit
def logpdf(
    theta,
    beta,
    model_d1, model_d2,
    model_d3, model_d4,
    noise,
    num_features,
    lengthscales,
    coefs,
    time_steps,
    zs,
    trans_epsilons,
    omegas,
    phis
):
    
    def body(prob, t):
        trans_eps = trans_epsilons[t - 1]
        omega = omegas[t - 1]
        phi = phis[t - 1]

        action = linear_policy(zs[t - 1], theta)
        model_input = jnp.stack([zs[t - 1], jnp.full((4,), action)]).T

        input = phi_X_batch(model_input, num_features, lengthscales, coefs, omega, phi)
    
        means = jnp.concatenate([model_d1[0], model_d2[0], model_d3[0], model_d4[0]])
        covs = jnp.diag(jnp.concatenate([
            jnp.diag(model_d1[1]),
            jnp.diag(model_d2[1]),
            jnp.diag(model_d3[1]),
            jnp.diag(model_d4[1])
        ]))
        d1, d2, d3, d4 = predict(means, covs, beta, input.reshape(-1), trans_eps)


        next_mean = jnp.array([d1, d2, d3, d4]) + zs[t - 1]

        prob += jax.scipy.stats.multivariate_normal.logpdf(
            zs[t],
            mean=next_mean,
            cov=jnp.eye(4) * (noise ** 2),
        )

        return prob, None
    

    init = 0
    logpdf_zs, _ = jax.lax.scan(body, init, time_steps)

    return logpdf_zs


def get_sequence_rewards(states):
    rewards = jnp.array([cost_function(state) for state in states])
    return rewards.sum()


def lr_gradients(
    theta,
    beta,
    model_d1, model_d2,
    model_d3, model_d4,
    horizon,
    num_features,
    lengthscales,
    coefs,
    noise,
    zs,
    trans_epsilons,
    omegas,
    phis,
):
    time_steps = jnp.arange(1, horizon)
    grad_func = vmap(
            vmap(
                jacobian(logpdf, 0),
                (
                    None, None, None, None, None, None, None, 
                    None, None, None, None, 0, None, 0, 0
                )
            ),
            (
                None, None, None, None, None, None, None, 
                None, None, None, None, 0, 0, None, None
            )

    )
    logpdf_grads = grad_func(
        theta,
        beta,
        model_d1, model_d2,
        model_d3, model_d4,
        noise,
        num_features,
        lengthscales,
        coefs,
        time_steps,
        zs,
        trans_epsilons,
        omegas,
        phis,
    )

    rewards = vmap(vmap(get_sequence_rewards, (0,)), (0,))(zs)

    lr_grads = logpdf_grads * rewards[:, :, jnp.newaxis]

    return lr_grads.mean(axis=(1, 0))

