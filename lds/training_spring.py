import optax
import jax.random as jrandom
import numpy as np

from constants import RAND_KEY
from spring_gradients import lr_gradients, rp_gradients
from spring_gradients import marginal_likelihood
from spring_utils import get_zs

def step(
    params,
    opt_state,
    optimizer,
    lr_estimator,
    mu0, V0,
    trans_noise,
    obs_noise,
    xs,
    num_steps,
    N,
    key
):
    zs, epsilons = get_zs(
        A=params[0],
        mu0=mu0, V0=V0,
        trans_noise=trans_noise,
        num_steps=num_steps,
        N=N,
        key=key,
    )

    if lr_estimator:
        # TODO
        grads = lr_gradients(params[0], mu0, V0, trans_noise, obs_noise, zs, xs)
    else:
        # TODO
        grads = rp_gradients(params[0], mu0, V0, trans_noise, obs_noise, epsilons, xs)

    objective_value = marginal_likelihood(
        params[0], mu0, V0, trans_noise, obs_noise, epsilons, xs
    )
    
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, objective_value, grads

def fit(
    params,
    optimizer,
    training_steps,
    mu0, V0,
    trans_noise,
    obs_noise, xs,
    num_steps,
    N,
    lr_estimator,
    key=RAND_KEY
):
    opt_state = optimizer.init(params)
    training_objectives = np.zeros((training_steps,))
    grad_values = np.zeros((training_steps, 2, 2))
 
    for i in range(training_steps):
        key, subkey = jrandom.split(key)
        params, opt_state, objective_value, grads = step(
            params=params,
            opt_state=opt_state, 
            optimizer=optimizer,
            lr_estimator=lr_estimator,
            mu0=mu0, V0=V0,
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
