import numpy as np


class EpsGreedyQPolicy():
    def __init__(self, epsilon=.1, decay_rate=1, min_epsilon=.01):
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon

    def select_action(self, q_values):
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.epsilon:
            action = np.random.random_integers(0, nb_actions - 1)
        else:
            action = np.argmax(q_values)

        return action

    def select_greedy_action(self, q_values):
        assert q_values.ndim == 1
        action = np.argmax(q_values)

        return action

    def update_epsilon(self):
        self.epsilon = self.epsilon * self.decay_rate
        if self.epsilon < self.min_epsilon:
            self.epsilon = self.min_epsilon
