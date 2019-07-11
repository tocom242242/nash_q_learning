import numpy as np

class MatrixGame():
    def __init__(self):
        self.reward_matrix = self._create_reward_table()

    def step(self, action1, action2):
        r1 = self.reward_matrix[0][action1][action2]
        r2 = self.reward_matrix[1][action1][action2]

        return None, r1, r2

    def _create_reward_table(self):
        reward_matrix = [
                            [[1, -1], [-1, 1]],
                            [[-1, 1], [1, -1]]
                        ]

        return reward_matrix
