import haiku as hk
import jax.numpy as jnp

from constants import SEED


def FeedForward(x):
    mlp = hk.nets.MLP(output_sizes=[256, 1])
    return mlp(x)

policy_net = hk.transform(FeedForward)
params = policy_net.init(SEED, jnp.ones((5, 4), dtype=jnp.float32))

def get_params():
    return params

def nn_policy(state, weights):
    preds = policy_net.apply(weights, None, state).squeeze()
    preds = jnp.tanh(preds)
    return preds