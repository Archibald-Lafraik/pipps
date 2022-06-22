import jax.random as jrandom

N = 1000 # Number of samples per point
NUM_INPUTS = 1000 # Number of input points in range
NUM_BATCHES = 10 # Number of estimators use to calculate estimator variance

RANGE_START = -10
RANGE_END = 10

# 0.09
VARIANCE = 0.09 # Sampling distribution variance

# JAX utils
RAND_KEY = jrandom.PRNGKey(1)