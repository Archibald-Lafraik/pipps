from jax import jacobian, vmap, jit
import jax.lax
import jax.numpy as jnp
import jax.scipy
import numpy as np

@jit
def likelihood(A, mu0, V0, trans_noise, obs_noise, epsilons, xs):
    L0 = jnp.linalg.cholesky(V0)
    L_trans = jnp.linalg.cholesky(trans_noise)

    z = mu0 + L0 @ epsilons[0]
    prob_x = jax.scipy.stats.multivariate_normal.pdf(
        xs[0], mean=z, cov=obs_noise
    )

    def body(carry, idx):
        z, prob_xs = carry
        z = A @ z + L_trans @ epsilons[idx]
        prob_xs *= jax.scipy.stats.multivariate_normal.pdf(
            xs[idx], mean=z, cov=obs_noise
        )
        return (z, prob_xs), None

    init = (z, prob_x)
    indices = jnp.arange(1, xs.shape[0])
    carry, _ = jax.lax.scan(body, init, indices)

    _, prob_xs = carry
    return prob_xs

def marginal_likelihood(A, mu0, V0, trans_noise, obs_noise, epsilons, xs):
    marg_likelihood = vmap(likelihood, in_axes=(None, None, None, None, None, 1, 1))
    return marg_likelihood(A, mu0, V0, trans_noise, obs_noise, epsilons, xs).mean(axis=0)


############################ Reparameterized Gradients ##########################

def rp_gradients(A, mu0, V0, trans_noise, obs_noise, epsilons, xs):
    grad_func = vmap(jacobian(likelihood, 0), in_axes=(None, None, None, None, None, 1, 1))
    grads = grad_func(A, mu0, V0, trans_noise, obs_noise, epsilons, xs)
    return grads.mean(axis=0)


############################ Likelihood-ratio Gradients ##########################

@jit
def lr_likelihood(obs_noise, zs, xs):
    init = 1

    def body(likelihood, idx):
        likelihood *= jax.scipy.stats.multivariate_normal.pdf(
            xs[idx], mean=zs[idx], cov=obs_noise
        )
        return likelihood, None

    indices = jnp.arange(1, zs.shape[0])
    likelihood, _ = jax.lax.scan(body, init, indices)

    return likelihood

@jit
def logpdf(A, mu0, V0, trans_noise, zs):
    init = jax.scipy.stats.multivariate_normal.logpdf(
        zs[0], mean=mu0, cov=V0
    )
    
    def body(logpdf, idx):
        logpdf += jax.scipy.stats.multivariate_normal.logpdf(
            zs[idx], mean=A @ zs[idx - 1], cov=trans_noise
        )
        return logpdf, None

    indices = jnp.arange(1, zs.shape[0])
    logpdf, _ = jax.lax.scan(body, init, indices)

    return logpdf



def lr_gradients(A, mu0, V0, trans_noise, obs_noise, zs, xs):
    grad_func = vmap(jacobian(logpdf, 0), in_axes=(None, None, None, None, 1))
    grads = grad_func(A, mu0, V0, trans_noise, zs)

    likelihood = vmap(lr_likelihood, in_axes=(None, 1, 1))(obs_noise, zs, xs)

    lr_grads = likelihood[:, np.newaxis, np.newaxis] * grads

    return lr_grads.mean(axis=0)
