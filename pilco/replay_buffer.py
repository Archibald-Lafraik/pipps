import numpy as np
import random
from collections import deque, namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state')
                    )    

class ReplayBuffer:

    def __init__(self, capacity):
        self.memory = []

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def __len__(self):
        return len(self.memory)

    def get_train_test_arrays(self, train_only=True, train_test_split=0.8):
        mem_size = len(self.memory)
        train_size = int(mem_size * train_test_split)
        random.shuffle(self.memory)

        states = np.array(list(map(lambda t: t.state, self.memory)))
        actions = np.array(list(map(lambda t: t.action, self.memory)))
        next_states = np.array(list(map(lambda t: t.next_state, self.memory)))

        inputs = np.zeros((mem_size, states.shape[1] * 2))
        for i in range(states.shape[1]):
            inputs[:, i] = states[:, i]
            inputs[:, i + 1] = actions

        if not train_only:
            train_X = inputs[:train_size]
            train_y = next_states[:train_size] - states[:train_size]

            test_X = inputs[train_size:]
            test_y = next_states[train_size:] - states[train_size:]
            return train_X, train_y, test_X, test_y
        else:
            y = next_states - states
            return inputs, y

