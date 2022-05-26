import jax.numpy as jnp
import jax.random as jrandom
import jax.nn
import optax
from jax import jit, vmap, grad

SEED = jrandom.PRNGKey(1)

def forward(model, weights, x):
    out = model.apply(weights, SEED, x)
    return jax.nn.softmax(out, axis=0)

def loss(params, model, input_data, actions, expected_returns):
    preds = forward(model, params, input_data)
    probs = jnp.take_along_axis(preds, indices=actions.reshape(-1, 1), axis=1).squeeze()
    return - jnp.sum(jnp.log(probs) * jnp.array(expected_returns))

@jit
def update(model, states, actions, returns, optimizer):
    grads = grad(loss, 0)(params, model, states, actions, returns) 
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)