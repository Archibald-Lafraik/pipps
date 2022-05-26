import jax.numpy as jnp
import torch
import jax.random as jrandom
import haiku as hk
import numpy as np
import optax
from jax import jit, grad

from policy_network import forward, loss
from utils import normalize_probs

ACTIONS = jnp.array([0, 1])
SEED = jrandom.PRNGKey(2)

output_sizes = [256, 2]
init_data = jnp.array([(0., 1., 2., 3.)for _ in range(5)])

def get_MLP(x):
    mlp = hk.nets.MLP(output_sizes=output_sizes)
    return mlp(x)

model = hk.transform(get_MLP)
initial_params = model.init(SEED, init_data)

optimizer = optax.chain(
    optax.adam(learning_rate=0.002),
    # optax.scale(-1.0)
)

@jit
def update(parameters, opt_state, states, actions, returns):
    grads = grad(loss, 0)(parameters, model, states, actions, returns) 
    updates, opt_state = optimizer.update(grads, opt_state, parameters)
    parameters = optax.apply_updates(parameters, updates)
    return parameters, opt_state, grads


def step(
    params,
    opt_state,
    # optimizer,
    env,
    horizon,
    gamma,
    # model,
    batch_size,
):
    cur_state = env.reset()
    # cur_state = env.reset()
    done = False
    transitions = []

    # Roll-out episode
    for t in range(horizon):
        act_prob = forward(model, params, cur_state)
        # act_prob = model(torch.from_numpy(cur_state)).float().data.numpy()
        act_prob = normalize_probs(act_prob)

        action = np.random.choice(ACTIONS, p=act_prob)
        prev_state = cur_state

        cur_state, _, done, info = env.step(action)
        cur_state = cur_state

        transitions.append((prev_state, action, t+1))
        # transitions.append((prev_state, action, t+1))
        if done:
            break
    score = len(transitions)

    state_batch, action_batch, reward_batch, = zip(*transitions)
    reward_batch = jnp.flip(jnp.array(reward_batch), axis=0)
    state_batch = jnp.array(state_batch)
    action_batch = jnp.array(action_batch)


    cumul_rewards = [(gamma ** (len(transitions) - 1)) * reward_batch[len(transitions) - 1]]
    for i in range(len(transitions) - 2, -1, -1):
        g = gamma ** (i) * reward_batch[i]
        cumul_rewards.append(cumul_rewards[-1] + g)
    cumul_rewards = jnp.flip(jnp.array(cumul_rewards), axis=0)
    expected_returns = cumul_rewards / cumul_rewards.max()

    params, opt_state, grads = update(
        params, opt_state, state_batch, action_batch, expected_returns
    )

    return params, opt_state, score, grads


def fit(
    env,
    # params,
    # optimizer,
    num_episodes,
    horizon,
    gamma,
    # model,
    batch_size,
):
    params = initial_params
    opt_state = optimizer.init(params)
    # opt_state = None 
    scores = np.zeros((num_episodes,))
    grad_values = np.zeros((num_episodes, len(params)))

    for i in range(num_episodes):
        params, opt_state, score, grads = step(
            params=params,
            opt_state=opt_state,
            # optimizer=optimizer,
            env=env,
            horizon=horizon,
            gamma=gamma,
            # model=model,
            batch_size=batch_size,
        )

        scores[i] = score
        # grad_values[i] = grads

        if i % 50 == 0:
            print(f"Episode {i}, Average Score: {jnp.mean(scores[:i])}")

    return params, scores, grad_values