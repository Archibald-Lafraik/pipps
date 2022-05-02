import optax
import numpy as np
from reparameterization import get_rp_gradients

from sampling_utils import get_z_samples
from likelihoodratio import get_likelihood_ratio_gradient


def loss(A_pred, A):
    return (A_pred - A) ** 2

def step(
    params,
    opt_state,
    optimizer,
    lr_estimator,
    mu0, V0, A, B, C, E, xs,
    num_inputs,
    N
):
    zs, epsilons = get_z_samples(num_inputs, N, mu0, V0, params[0], B)
    
    if lr_estimator:
        grads = get_likelihood_ratio_gradient(
            A=params[0],
            B=B, C=C, E=E, zs=zs, xs=xs,
            num_inputs=num_inputs,
            N=N
        )
    else:
        grads = get_rp_gradients(
            mu0=mu0,
            V0=V0,
            A=params[0],
            B=B, C=C, E=E, xs=xs,
            epsilons=epsilons
        )
    # How should gradients be aggregated?
    grads = grads.mean()
    loss_value = loss(params[0], A)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss_value, grads

def fit(
    params,
    optimizer,
    training_steps,
    mu0, V0, A, B, C, E, xs,
    num_inputs,
    N,
    lr_estimator,
):
    opt_state = optimizer.init(params)
    training_losses = np.zeros((training_steps,))
    grad_values = np.zeros((training_steps,))
 
    for i in range(training_steps):
        params, opt_state, loss_value, grads = step(
            params, opt_state, optimizer, lr_estimator, mu0, V0, A, B, C, E, xs, num_inputs, N
        )
        training_losses[i] = loss_value
        grad_values[i] = grads
        if i % 100 == 0:
            print(f'Step {i}, loss: {loss_value:5f}, grad: {grads:4f}, A - {params[0]:5f}')

    return params, training_losses, grad_values
