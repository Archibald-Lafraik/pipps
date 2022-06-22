import optax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

import matplotlib.pyplot as plt

from constants import RAND_KEY
from rff_gradients import lr_gradients, rp_gradients
from rff import get_zs, elbo

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
    key,
    step
):
    subkey = jrandom.split(key, num=3)
    epsilons = jrandom.normal(key=subkey[0], shape=(num_steps, N, xs.shape[-1]))
    omegas = jrandom.normal(key=subkey[1], shape=(num_steps, num_features, N, xs.shape[-1]))
    phis = jrandom.uniform(key=subkey[2], minval=0, maxval=2 * jnp.pi, shape=(num_steps, num_features, N, 1))
    
    zs = get_zs(
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
            obs_noise, start_state, V0, omegas, phis, zs, epsilons
        )
    else:
        grads = rp_gradients(
            params[0], xs, num_features, lengthscale, coef, trans_noise,
            obs_noise, start_state, V0, omegas, phis, epsilons
        )

    objective_value = elbo(
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
    grad_values = np.zeros((training_steps, num_features, xs.shape[-1]))
 
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
            key=subkey,
            step=i
        )
        training_objectives[i] = objective_value
        grad_values[i] = grads

        if i % 100 == 0:
            print(f'Step {i}, marginal likelihood: {objective_value:5f}, A - {params[0]}')

    return params, training_objectives, grad_values
