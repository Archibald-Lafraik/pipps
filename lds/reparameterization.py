import jax.scipy
import jax.lax
import jax.numpy as jnp
from jax import vmap, grad, jit

def marginal_likelihood(A, B, C, E, x, z_prev, eps):
    z = A * z_prev + jnp.sqrt(B) * eps
    return jax.scipy.stats.norm.pdf(x, loc=C * z, scale=jnp.sqrt(E))

@jit
def scan_grad(mu0, V0, A, B, C, E, xs, epsilons):
    def body(z, x_eps):
        x, eps = x_eps
        my_grad = grad(marginal_likelihood, 0)(A, B, C, E, x, z, eps)
        return A * z + jnp.sqrt(B) * eps, my_grad

    init = mu0 + epsilons[0] * jnp.sqrt(V0)
    xs_eps = jnp.array(list(zip(xs[1:], epsilons[1:])))
    _, scan_grads = jax.lax.scan(body, init, xs_eps)

    xs_grads = jnp.zeros((xs.shape[0],))
    xs_grads = xs_grads.at[0].set(0)     # Because p(X0|z0) independent of A
    xs_grads = xs_grads.at[1:].set(scan_grads)

    return xs_grads

def get_rp_gradients(mu0, V0, A, B, C, E, xs, epsilons):
    func = vmap(scan_grad, in_axes=(None, None, None, None, None, None, 1, 1))
    return func(mu0, V0, A, B, C, E, xs, epsilons).mean(axis=0)