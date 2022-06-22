from jax import vmap, jacobian
import jax.numpy as jnp

from kink import logpdf_z

def loss(
    theta,
    start_state,
    V0,
    trans_noise,
    zs, 
):
    prob_zs = vmap(logpdf_z, (None, None, None, None, 1))(theta, start_state, V0, trans_noise, zs)
    return prob_zs.mean(axis=0)

def loss_grad(
    theta,
    start_state,
    V0,
    trans_noise,
    zs, 
):
    grads = vmap(jacobian(logpdf_z, 0), (None, None, None, None, 1))(theta, start_state, V0, trans_noise, zs)
    grads = grads.squeeze()
    return grads.mean(axis=0) 


def clip_grads(grads):
    for i in range(grads.shape[0]):
        if grads[i] < -10e4:
            grads = grads.at[i].set(-10e4)
        elif grads[i] > 10e4:
            grads = grads.at[i].set(10e4)
    return grads