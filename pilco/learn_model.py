import jax.numpy as jnp
import jax.scipy
import jax.lax
import jax.random as jrandom
from jax import jit, vmap, jacobian

from rff import phi_X_batch, phi_X
from trans_model import posterior, predict

def train_transition_models(X, y, beta, model, num_features, lengthscales, coefs, key):
    omega = jrandom.normal(key=key, shape=(num_features, 1))
    phi = jrandom.uniform(key=key, minval=0, maxval=2 * jnp.pi, shape=(num_features, 1))

    y = y - X
    X = X[:, jnp.newaxis]
    model = rff_posterior(
        num_features, lengthscales, coefs, omega, phi, beta, X, y, model
    ) 
    
    return model

@jit
def rff_posterior(
    num_features,
    lengthscales,
    coefs,
    omega,
    phi,
    beta,
    X, y,
    model
):
    phi_X = phi_X_batch(X, num_features, lengthscales, coefs, omega, phi).T
    # phi_X = X

    model = posterior(*model, beta, phi_X, y)

    return model

@jit
def pred(
    start_state,
    trans_eps, state_eps,
    omega, phi,
    model,
    num_features,
    lengthscales,
    coefs,
    beta,
    model_noise
):
    input = phi_X(start_state, num_features, lengthscales, coefs, omega, phi)
    # input = start_state

    diff = predict(*model, beta, input, trans_eps)
    next_mean = diff + start_state
    next_state = next_mean + state_eps * model_noise
    return next_state

def state_prediction(
    start_state,
    model,
    model_noise,
    num_features,
    lengthscales,
    coefs,
    beta,
    N,
    key
):
    keys = jrandom.split(key, num=4)
    state_eps = jrandom.normal(keys[0], shape=(N, 1))
    trans_eps = jrandom.normal(keys[1], shape=(N, 1))
    omega = jrandom.normal(keys[2], shape=(num_features, 1))
    phi = jrandom.uniform(keys[3], minval=0, maxval=2 * jnp.pi, shape=(num_features, 1))

    foo = vmap(pred, (None, 0, 0, None, None, None, None, None, None, None, None))
    predictions = foo(
        start_state, trans_eps, state_eps, omega, phi, model, num_features,
        lengthscales, coefs, beta, model_noise
    )

    return predictions


@jit
def get_likelihood(
    start_states,
    obs_states,
    trans_epsilons,
    omega, phi,
    model,
    num_features,
    lengthscales,
    coefs,
    beta,
    indices,
    model_noise
):
    def body(prob, idx):
        start_state = jnp.array([start_states[idx]])
        trans_eps = trans_epsilons[idx]


        input = phi_X(start_state, num_features, lengthscales, coefs, omega, phi)
        diff = predict(*model, beta, input, trans_eps)
        next_mean = (diff + start_state).squeeze()
        prob += jax.scipy.stats.norm.logpdf(
            obs_states[idx], loc=next_mean, scale=model_noise
        )
        return prob, None

    lkhd, _ = jax.lax.scan(body, 0, indices)
    return - lkhd
    
def grad_marg_lkhd(
    start_states,
    obs_states,
    model,
    num_features,
    lengthscales,
    coefs,
    beta,
    indices,
    model_noise,
    grad_position,
    N,
    key
):
    keys = jrandom.split(key, num=4)
    horizon = obs_states.shape[0]
    trans_eps = jrandom.normal(keys[1], shape=(N, horizon, 1))
    omega = jrandom.normal(keys[2], shape=(num_features, 1))
    phi = jrandom.uniform(keys[3], minval=0, maxval=2 * jnp.pi, shape=(num_features, 1))

    lkhd_func = vmap(
        jacobian(get_likelihood, grad_position),
        (None, None, 0, None, None, None, None, None, None, None, None, None)
    )
    grads = lkhd_func(
        start_states, obs_states, trans_eps, omega, phi, model, 
        num_features, lengthscales, coefs, beta, indices, model_noise
    )

    return grads.mean(axis=0)

def marginal_likelihood(
    start_states,
    obs_states,
    model,
    num_features,
    lengthscales,
    coefs,
    beta,
    indices,
    model_noise,
    N,
    key
):
    keys = jrandom.split(key, num=4)
    horizon = obs_states.shape[0]
    trans_eps = jrandom.normal(keys[1], shape=(N, horizon, 1))
    omega = jrandom.normal(keys[2], shape=(N, num_features, 1))
    phi = jrandom.uniform(keys[3], minval=0, maxval=2 * jnp.pi, shape=(N, num_features, 1))

    lkhd_func = vmap(
        get_likelihood,
        (None, None, 0, 0, 0, None, None, None, None, None, None, None)
    )
    lkhds = lkhd_func(
        start_states, obs_states, trans_eps, omega, phi, model, 
        num_features, lengthscales, coefs, beta, indices, model_noise
    )

    return lkhds.mean(axis=0)