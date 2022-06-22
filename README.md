# Monte-Carlo Gradient Estimators for Chaotic Systems

This project aims to investigate the variance of the likelihood-ratio and reparameterized gradient estimators in chaotic systems. This was motivated by surprising claims about the exploding gradient variance of reparameterized gradients in model-based reinforcement learning, published in "PIPPS: Flexible Model-Based Policy Search Robust to the Curse of Chaos", which you can find [here](https://arxiv.org/pdf/1902.01240.pdf). These claims clashed with previously established results from stochastic variational inference, which stated that reparameterized gradient estimators had much lower variance than the score gradients.

This repo contains the code of different experiments, which help investigate these claims in various settings of increasing complexity. It also provides efficient implementations of the likelihood-ratio and reparameterized gradient estimators in JAX.

# Table of Contents

- [Experiments](#experiments)
  * [One-Dimensional Parameterized Function](#1d)
  * [Dynamical Systems](#ds)
  * [Model-Based Reinforcement Learning](#mbrl)
- [Contributing](#contributing)
- [Authors](#authors)

# Experiments <a name="experiments"></a>

This section describes the different experiments performed and where the corresponding code can be found to replicate experiments.

## One-Dimensional Parameterized Function <a name="1d"></a>

The first experiments were done using a convolved one-dimensional parameterized function. In this setting, we showed that the reparameterized gradient estimator variance increases exponentially with the 'level of chaos' of the function, whereas the variance of likelihood-ratio gradients is constant, irrespective of the level of chaos. To reproduce these experiments, you can run the `toy_experiment.ipynb` notebook.

This notebook also contains three testing techniques for stochastic gradient estimators:
- Comparison to a closed-form solution
- Consistency checks between different gradient estimators
- Comparison of the variance decrease rate (as a function of the number of samples) to the theoretical Monte-Carlo error decrease rate

The code used for these experiments is contained in the `initial_experiments` folder.

## Dynamical systems <a name="ds"></a>

In these experiments, we use the gradient estimators to learn the transition function of dynamical systems, as many reinforcement learning control tasks take place in in such settings. The chosen dynamical systems were the mass-spring system, which is linear, and the double pendulum, which is non-linear and chaotic. 

- The dynamics of the mass-spring system were successfully learned by both estimators. To reproduce this experiment, you can run the `spring_system.ipynb` notebook from the `dynamical_systems` folder.
- Unfortunately, the dynamics of the double pendulum could not be learned by either gradient estimators, preventing us from comparing their variance in this chaotic environment. This is attributed to the system's high sensitivity to initial conditions, causing the marginal likelihood objective and its gradient to vanish to zero. To reproduce this experiment, you can run the `double_pendu_main.ipynb` notebook in the `double_pendulum` folder.

## Model-based Reinforcement Learning <a name="mbrl"></a>

The final set of experiments were performed in the MuJoCo control task suite for reinforcement learning. This was the opportunity to implement a particle-based version of PILCO using the likelihood-ratio and reparameterized gradients, to replicate the experiments done in PIPPS and compare the results. The first experiment we applied our algorithm to is the inverted pendulum balancing task. This task was solved by both gradient estimators, with the likelihood-ratio gradients version converging slightly slower. The gradient variance of the score gradients was 100 times higher than the reparameterized gradients' in this non-chaotic environment, which is expected.

To replicate this experiment, head to the `model_based_rl` folder and run the `inv_pendu_main.ipynb` notebook.

The implementations of the likelihood-ratio and reparameterized gradient estimators for model-based reinforcement learning can be found in the `policy.py` file. From personal experimentation, I find them to be much faster than the Optax (optimisation library for JAX) implementations.


# Contributing <a name="contributing"></a>

This project is far from being finished, and is open to a broad range of contributions! Please feel free to submit your ideas in the `Issues` tab :)

Here are some important next steps I identified:
- Applying the particle-based version of PILCO to more complex control tasks, where the curse of chaos can be observed
- Implementation the Total Propagation algorithm from PIPPS using jAX
- Experiment with new transition models for the particle-based PILCO. I am especially curious about the performance of probabilistic ensembles of neural networks.

# Authors <a name="authors"></a>

**Archibald Fraikin** - [LinkedIn](https://www.linkedin.com/in/archibald-fraikin-819607194/)

This project was supervised by [**Dr. Mark Van der Wilk**](https://markvdw.github.io/)


