import haiku as hk
import jax.numpy as jnp


def FeedForward(x):
    mlp = hk.nets.MLP(output_sizes=[256, 1])
    return mlp(x)

policy_net = hk.without_apply_rng(hk.transform(FeedForward))

def get_params(key):
    params = policy_net.init(key, jnp.ones((5, 4), dtype=jnp.float32))
    return params

def nn_policy(state, weights):
    preds = policy_net.apply(params=weights, x=state).squeeze()
    preds = jnp.tanh(preds)
    return preds