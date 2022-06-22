import jax.scipy
import jax.lax
import jax.numpy as jnp
import jax.random as jrandom
from jax import jacobian, jit, vmap

from rff import phi_X_batch, phi_X

def prior(num_features, alpha):
    mean = jnp.zeros(num_features)
    cov_inv = jnp.eye(num_features) / alpha

    return mean, cov_inv

def posterior(mean, cov_inv, beta, phiX, y):
    cov_inv = cov_inv + beta * (phiX.T @ phiX)
    # TODO consider replacing by more efficient inverse method
    # (For now no speed issues)
    cov = jnp.linalg.inv(cov_inv)

    mean = cov @ (cov_inv @ mean + beta * y @ phiX)

    return mean, cov

@jit
def trans_output(w_d1, w_d2, w_d3, w_d4, x):
    d1 = w_d1 @ x[0]
    d2 = w_d2 @ x[1]
    d3 = w_d3 @ x[2]
    d4 = w_d4 @ x[3]
    return jnp.stack([d1, d2, d3, d4])

@jit
def predict(mean, cov, beta, phi_Xstar, eps):
    y_pred_mean = phi_Xstar @ mean
    # print(f"Old technique: {y_pred_mean}")
    y_pred_var = 1 / beta + phi_Xstar @ cov @ phi_Xstar.T
    
    sample = y_pred_mean + eps * jnp.sqrt(y_pred_var)
    
    return sample

@jit
def predict_params(mean, cov, beta, phi_Xstar, eps):
    y_pred_mean = phi_Xstar @ mean
    y_pred_var = 1 / beta + phi_Xstar @ cov @ phi_Xstar.T
    
    sample = y_pred_mean + eps * jnp.sqrt(y_pred_var)
    
    return y_pred_mean, y_pred_var, sample

@jit
def predict_batch(mean, cov, beta, phi_Xstar, eps):
    y_pred_mean = phi_Xstar.T @ mean
    w_cov = cov
    y_pred_var = 1 / beta + (phi_Xstar.T @ w_cov * phi_Xstar).sum(axis=1)

    sample = y_pred_mean.squeeze() + eps * jnp.sqrt(y_pred_var.squeeze())
    
    return sample


def train_transition_models(replay_buffer, betas, trans_models, num_features):
    X, y = replay_buffer.get_train_test_arrays()

    X_d1 = X[:, 0]
    X_d2 = X[:, 1]
    X_d3 = X[:, 2]
    X_d4 = X[:, 3]
    y_d1 = y[:, 0]
    y_d2 = y[:, 1]
    y_d3 = y[:, 2]
    y_d4 = y[:, 3]

    model_d1, model_d2, model_d3, model_d4 = rff_posterior(
        betas,
        X_d1, X_d2, X_d3, X_d4, y_d1, y_d2, y_d3, y_d4,
        *trans_models
    ) 
    
    return [model_d1, model_d2, model_d3, model_d4]

@jit
def rff_posterior(
    betas,
    X_d1, X_d2,
    X_d3, X_d4,
    y_d1, y_d2,
    y_d3, y_d4,
    model_d1, model_d2,
    model_d3, model_d4
):
    model_d1 = posterior(*model_d1, betas[0], X_d1, y_d1)
    model_d2 = posterior(*model_d2, betas[1], X_d2, y_d2)
    model_d3 = posterior(*model_d3, betas[2], X_d3, y_d3)
    model_d4 = posterior(*model_d4, betas[3], X_d4, y_d4)

    return model_d1, model_d2, model_d3, model_d4


###################### Optimize model ######################

@jit
def likelihood(
    start_states,
    num_features,
    lengthscales,
    coefs,
    betas,
    model_noise,
    actions,
    obs_states,
    trans_epsilons,
    omega, phi,
    m_d1, m_d2,
    m_d3, m_d4,
    indices
):
    def body(prob, idx):
        trans_eps = trans_epsilons[idx]

        model_input = jnp.stack([start_states[idx], jnp.full((4,), actions[idx])]).T

        # in_d1 = phi_X(model_input[0, jnp.newaxis], num_features, lengthscales[0], coefs[0], omega, phi)
        # in_d2 = phi_X(model_input[1, jnp.newaxis], num_features, lengthscales[1], coefs[1], omega, phi)
        # in_d3 = phi_X(model_input[2, jnp.newaxis], num_features, lengthscales[2], coefs[2], omega, phi)
        # in_d4 = phi_X(model_input[3, jnp.newaxis], num_features, lengthscales[3], coefs[3], omega, phi)

        means = jnp.concatenate([m_d1[0], m_d2[0], m_d3[0], m_d4[0]])
        # covs = jnp.zeros((means.shape[0], means.shape[0]))
        # covs = covs.at[:1000, :1000].set(m_d1[1])
        # covs = covs.at[1000:1000 * 2, 1000:1000 * 2].set(m_d2[1])
        # covs = covs.at[1000 * 2:1000 * 3, 1000 * 2:1000 * 3].set(m_d3[1])
        # covs = covs.at[1000 * 3:1000 * 4, 1000 * 3:1000 * 4].set(m_d4[1])
        # d1, d2, d3, d4 = predict(means, covs, betas, jnp.concatenate([in_d1, in_d2, in_d3, in_d4]), trans_eps)
        covs = jnp.zeros((means.shape[0], means.shape[0]))
        covs = covs.at[:, :2].set(m_d1[1])
        covs = covs.at[2:2 * 2, 2:2 * 2].set(m_d2[1])
        covs = covs.at[2 * 2:2 * 3, 2 * 2:2 * 3].set(m_d3[1])
        covs = covs.at[2 * 3:2 * 4, 2 * 3:2 * 4].set(m_d4[1])
        d1, d2, d3, d4 = predict(means, covs, betas, model_input, trans_eps)

        next_mean = jnp.array([d1, d2, d3, d4]) + start_states[idx]
        prob += jax.scipy.stats.multivariate_normal.logpdf(
            obs_states[idx], mean=next_mean, cov=jnp.eye(4) * (model_noise ** 2)
        )
        return prob, None

    prob, _ = jax.lax.scan(body, 0, indices)
    
    return - 1 * prob

def lklhood_grad(
    start_states,
    num_features,
    lengthscales,
    coefs,
    betas,
    model_noise,
    grad_position,
    actions,
    obs_states,
    trans_eps,
    omegas, phis,
    m_d1, m_d2,
    m_d3, m_d4
):
    marg_lkhd_func = vmap(
        jacobian(likelihood, grad_position),
        (None, None, None, None, None, None, None, None, 0, 0, 0, None, None, None, None, None)
    )
    indices = jnp.arange(start_states.shape[0])
    grads = marg_lkhd_func(
        start_states,
        num_features,
        lengthscales,
        coefs,
        betas,
        model_noise,
        actions,
        obs_states,
        trans_eps,
        omegas, phis,
        m_d1, m_d2,
        m_d3, m_d4,
        indices
    )

    return grads.mean(axis=0)

def marg_lklhood(
    start_states,
    num_features,
    lengthscales,
    coefs,
    betas,
    model_noise,
    actions,
    obs_states,
    trans_eps,
    omegas, phis,
    m_d1, m_d2,
    m_d3, m_d4
):
    marg_lkhd_func = vmap(
        likelihood,
        (None, None, None, None, None, None, None, None, 0, 0, 0, None, None, None, None, None)
    )
    indices = jnp.arange(start_states.shape[0])
    grads = marg_lkhd_func(
        start_states,
        num_features,
        lengthscales,
        coefs,
        betas,
        model_noise,
        actions,
        obs_states,
        trans_eps,
        omegas, phis,
        m_d1, m_d2,
        m_d3, m_d4,
        indices,
    )

    return grads.mean(axis=0)