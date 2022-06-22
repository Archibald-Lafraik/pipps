import jax.scipy
from jax import vmap, jit, jacobian
import jax.numpy as jnp
import jax.random as jrandom
from constants import RAND_KEY

@jit
def foo(x):
    return jnp.where(x < 4, x + 1,-4 * x + 21)


def get_state_sequence(
    start_state,
    V0,
    trans_noise,
    obs_noise, 
    num_steps, 
    N,
    key=RAND_KEY
):
    states = jnp.zeros((num_steps, N,))
    obs = jnp.zeros((num_steps, N,))  

    epsilons = jrandom.normal(key=key, shape=(num_steps, N,))

    z = start_state + epsilons[0] * jnp.sqrt(V0)
    cur_obs = z + epsilons[0] * jnp.sqrt(obs_noise)

    states = states.at[0].set(z)
    obs = obs.at[0].set(cur_obs)

    for i in range(1, num_steps):
        next_state = vmap(foo, (0,))(z)
        z = next_state + epsilons[i] * jnp.sqrt(trans_noise)
        cur_obs = z + epsilons[i] * jnp.sqrt(obs_noise)

        states = states.at[i].set(z)
        obs = obs.at[i].set(cur_obs) 

    return states, obs

############################# MODEL PREDICTIONS #################################

def forward(theta, x):
    num_features = theta.shape[0]
    features = jnp.array([
        x ** i for i in range(num_features)
    ])
    return features.T @ theta


def predict(
    theta,
    inputs
):
    predictions = vmap(forward, (None, 1), 1)(theta, inputs)
    return predictions.T

################################ STATE SEQUENCES #############################

def get_zs(
    theta,
    start_state,
    V0,
    trans_noise,
    epsilons,
):
    zs_func = vmap(
        get_z_seq,
        (None, None, None, None, 1),
        1
    )
    return zs_func(theta, start_state, V0, trans_noise, epsilons)

@jit
def get_z_seq(
    theta,
    start_state,
    V0,
    trans_noise,
    epsilons
):
    zs = jnp.zeros_like(epsilons)

    z = start_state + epsilons[0] * jnp.sqrt(V0)
    zs = zs.at[0].set(z)

    for i in range(1, epsilons.shape[0]):
        pred_state = forward(theta, z)
        z = pred_state + epsilons[i] * jnp.sqrt(trans_noise)
        zs = zs.at[i].set(z)
    
    return zs


@jit
def logpdf_z(
    theta,
    start_state,
    V0,
    trans_noise,
    zs,
):
    init = jax.scipy.stats.norm.logpdf(
        zs[0], loc=start_state, scale=jnp.sqrt(V0)
    )

    def body(logpdf, idx):
        predicted_state = forward(theta, zs[idx - 1])
        logpdf += jax.scipy.stats.norm.logpdf(
            zs[idx],
            loc=predicted_state,
            scale=jnp.sqrt(trans_noise)
        )
        return logpdf, None

    indices = jnp.arange(1, zs.shape[0])
    logpdf, _ = jax.lax.scan(body, init, indices)
    return logpdf

@jit
def log_likelihood(
    theta,
    start_state,
    V0,
    trans_noise,
    obs_noise,
    xs,
    epsilons,
):
    z = start_state + epsilons[0] * jnp.sqrt(V0)
    prob_xs = jax.scipy.stats.norm.logpdf(
        xs[0], loc=z, scale=jnp.sqrt(obs_noise)
    ) 

    def body(carry, idx):
        z, prob_xs = carry
        next_state = forward(theta, z)
        z = next_state + epsilons[idx] * jnp.sqrt(trans_noise)
        prob_xs += jax.scipy.stats.norm.logpdf(
            xs[idx], loc=z, scale=jnp.sqrt(obs_noise)
        )
        return (z, prob_xs), None

    init = (z, prob_xs)
    indices = jnp.arange(1, epsilons.shape[0])
    carry, _ = jax.lax.scan(body, init, indices)

    return carry[1]


@jit
def elbo(
    theta,
    start_state,
    V0,
    trans_noise,
    obs_noise,
    xs,
    epsilons,
):
    lklhood_func = vmap(
        log_likelihood,
        (None, None, None, None, None, 1, 1)
    )
    lklhood = lklhood_func(theta, start_state, V0, trans_noise, obs_noise, xs, epsilons)
    
    return lklhood.mean(axis=0)



################################# GRADIENTS ##################################

def rp_gradients(
    theta,
    start_state,
    V0,
    trans_noise,
    obs_noise,
    xs,
    epsilons,
):
    lklhood_func = vmap(
        jacobian(log_likelihood, 0),
        (None, None, None, None, None, 1, 1)
    )
    lklhood = lklhood_func(theta, start_state, V0, trans_noise, obs_noise, xs, epsilons)
    lklhood = lklhood.squeeze()
    
    # Gradient clipping
    # lklhood = jnp.nan_to_num(lklhood)

    return lklhood.mean(axis=0)
