import optax
import jax.random as jrandom
import numpy as np

from constants import RAND_KEY
from sampling_utils import get_z_samples
from reparameterization import get_rp_gradients, objective
from likelihoodratio import get_lr_gradients 

def step(
    params,
    opt_state,
    optimizer,
    lr_estimator,
    mu0, V0, B, C, E, xs,
    num_inputs,
    N,
    key
):
    zs, epsilons = get_z_samples(num_inputs=num_inputs, N=N, mu_0=mu0, V_0=V0, A=params[0], B=B, key=key)

    if lr_estimator:
        grads = get_lr_gradients(mu0=mu0, V0=V0, A=params[0], B=B, C=C, E=E, xs=xs, zs=zs)
    else:
        grads = get_rp_gradients(mu0=mu0, V0=V0, A=params[0], B=B, C=C, E=E, epsilons=epsilons, xs=xs)

    # How should gradients be aggregated?
    grads = grads.mean()
    objective_value = objective(mu0, V0, params[0], B, C, E, epsilons, xs)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, objective_value, grads

def fit(
    params,
    optimizer,
    training_steps,
    mu0, V0, B, C, E, xs,
    num_inputs,
    N,
    lr_estimator,
    key=RAND_KEY
):
    opt_state = optimizer.init(params)
    training_objectives = np.zeros((training_steps,))
    grad_values = np.zeros((training_steps,))
 
    for i in range(training_steps):
        key, subkey = jrandom.split(key)
        params, opt_state, objective_value, grads = step(
            params=params,
            opt_state=opt_state, 
            optimizer=optimizer,
            lr_estimator=lr_estimator,
            mu0=mu0, V0=V0, B=B, C=C, E=E, xs=xs,
            num_inputs=num_inputs,
            N=N,
            key=subkey
        )
        training_objectives[i] = objective_value
        grad_values[i] = grads

        if i % 100 == 0:
            print(f'Step {i}, marginal likelihood: {objective_value:5f}, grad: {grads:4f}, A - {params[0]:5f}')

    return params, training_objectives, grad_values
