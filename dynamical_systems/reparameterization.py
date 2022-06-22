import jax.scipy
import jax.numpy as jnp
from jax import grad

def objective(mu0, V0, A, B, C, E, epsilons, xs):
    z_prev = mu0 + jnp.sqrt(V0) * epsilons[0]
    joint_x = jax.scipy.stats.norm.pdf(xs[0], loc=C * z_prev, scale=jnp.sqrt(E))

    for i in range(1, epsilons.shape[0]):
        z = A * z_prev + jnp.sqrt(B) * epsilons[i]
        joint_x *= jax.scipy.stats.norm.pdf(xs[i], loc=C * z, scale=jnp.sqrt(E))
        z_prev = z

    return joint_x.mean()

def get_rp_gradients(mu0, V0, A, B, C, E, epsilons, xs):
    res = grad(objective, 2)(mu0, V0, A, B, C, E, epsilons, xs)

    return res