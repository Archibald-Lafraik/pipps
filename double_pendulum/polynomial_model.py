from jax import jit, vmap
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy

def forward(theta, x):
    num_features = theta.shape[0]
    features = jnp.stack([
        x ** i for i in range(num_features)
    ])
    return features.T @ theta


def predict(
    theta,
    inputs
):
    predictions = vmap(forward, (None, 1), 1)(theta, inputs)
    return predictions.T


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
    L0 = jnp.linalg.cholesky(V0)
    L_trans = jnp.linalg.cholesky(trans_noise)

    zs = jnp.zeros_like(epsilons)

    z = start_state + L0 @ epsilons[0]
    zs = zs.at[0].set(z)

    for i in range(1, epsilons.shape[0]):
        pred_state = forward(theta, z)
        z = pred_state + L_trans @ epsilons[i]
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
    init = jax.scipy.stats.multivariate_normal.logpdf(
        zs[0], mean=start_state, cov=V0
    )

    def body(logpdf, idx):
        predicted_state = forward(theta, zs[idx - 1])
        logpdf += jax.scipy.stats.multivariate_normal.logpdf(
            zs[idx],
            mean=predicted_state,
            cov=trans_noise
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
    L0 = jnp.linalg.cholesky(V0)
    L_trans = jnp.linalg.cholesky(trans_noise)

    z = start_state + L0 @ epsilons[0]
    prob_xs = jax.scipy.stats.multivariate_normal.logpdf(
        xs[0], mean=z, cov=obs_noise
    ) 

    def body(carry, idx):
        z, prob_xs = carry
        next_state = forward(theta, z)
        z = next_state + L_trans @ epsilons[idx]
        prob_xs += jax.scipy.stats.multivariate_normal.logpdf(
            xs[idx], mean=z, cov=obs_noise
        )
        return (z, prob_xs), None

    init = (z, prob_xs)
    indices = jnp.arange(1, epsilons.shape[0])
    carry, _ = jax.lax.scan(body, init, indices)

    return carry[1]



def marginal_likelihood(
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