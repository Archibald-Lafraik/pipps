import optax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from constants import RAND_KEY
from rff_gradients import lr_gradients, rp_gradients
from rff import get_zs, marginal_likelihood

def step(
    params,
    opt_state,
    optimizer,
    lr_estimator,
    num_features,
    lengthscale,
    coef, V0,
    start_state,
    trans_noise,
    obs_noise,
    xs,
    num_steps,
    N,
    key
):
    epsilons = jrandom.normal(key=key, shape=(num_features, N, 4))
    omegas = jrandom.normal(key=key, shape=(num_features, N, 1))
    phis = jrandom.uniform(key=key, minval=0, maxval=2 * jnp.pi, shape=(num_features, N, 1))
    
    zs = get_zs(
        num_steps=num_steps,
        N=N,
        theta=params[0],
        num_features=num_features,
        lengthscale=lengthscale,
        coef=coef,
        trans_noise=trans_noise,
        start_state=start_state, V0=V0,
        omegas=omegas, phis=phis,
        epsilons=epsilons,
    )

    if lr_estimator:
        grads = lr_gradients(
            params[0], xs, num_features, lengthscale, coef, trans_noise,
            obs_noise, start_state, V0, omegas, phis, zs
        )
    else:
        grads = rp_gradients(
            params[0], xs, num_features, lengthscale, coef, trans_noise,
            obs_noise, start_state, V0, omegas, phis, epsilons
        )

    objective_value = marginal_likelihood(
        params[0], xs, num_features, lengthscale, coef, trans_noise,
        obs_noise, start_state, V0, omegas, phis, epsilons
    )
    
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, objective_value, grads

def fit(
    params,
    optimizer,
    training_steps,
    num_features,
    lengthscale,
    coef, V0,
    start_state,
    trans_noise,
    obs_noise, xs,
    num_steps,
    N,
    lr_estimator,
    key=RAND_KEY
):
    opt_state = optimizer.init(params)
    training_objectives = np.zeros((training_steps,))
    grad_values = np.zeros((training_steps, 4))
 
    for i in range(training_steps):
        key, subkey = jrandom.split(key)
        params, opt_state, objective_value, grads = step(
            params=params,
            opt_state=opt_state, 
            optimizer=optimizer,
            lr_estimator=lr_estimator,
            num_features=num_features,
            lengthscale=lengthscale,
            coef=coef,
            start_state=start_state, V0=V0,
            trans_noise=trans_noise,
            obs_noise=obs_noise, xs=xs,
            num_steps=num_steps,
            N=N,
            key=subkey
        )
        training_objectives[i] = objective_value
        grad_values[i] = grads

        if i % 100 == 0:
            print(f'Step {i}, marginal likelihood: {objective_value:5f}, A - {params[0]}')

    return params, training_objectives, grad_values
