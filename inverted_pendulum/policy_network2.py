from jax import jit, vmap
import jax.nn
import jax.numpy as jnp
import jax.random as jrandom

SEED = jrandom.PRNGKey(2)

def initialize_weights(layer_sizes, key=SEED):
    # TODO make it return a jnp array?
    keys = jrandom.split(key, len(layer_sizes))

    def initialize_layer(m, n, key, scale=1e-2):
        w_key, b_key = jrandom.split(key)
        return scale * jrandom.normal(w_key, (n, m)), scale * jrandom.normal(b_key, (n,))
    
    layers = [initialize_layer(m, n, k) for m, n, k in zip(layer_sizes[:-1], layer_sizes[1:], keys)]
    return layers


def LinearLayer(weights, x):
    w, b = weights
    return jnp.dot(w, x) + b


def forward(params, in_data):
    def body(x):
        out = x

        for w, b in params[:-1]:
            out = LinearLayer([w, b], out)
            out = jax.nn.relu(out)
        
        w, b = params[-1]
        logits = jnp.dot(w, out) + b
        return jax.nn.softmax(logits, axis=0) 

    return vmap(body, (0,))(in_data)
        

def loss(params, input_data, targets, expected_returns):
    preds = forward(params, input_data)
    probs = jnp.take_along_axis(preds, indices=targets.reshape(-1, 1), axis=1).squeeze()
    return - jnp.sum(jnp.log(probs) * jnp.array(expected_returns))
