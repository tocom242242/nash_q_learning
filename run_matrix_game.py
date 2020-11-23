import numpy as np
import matplotlib.pyplot as plt
from nash_q_learner import NashQLearner
from matrix_game import MatrixGame

if __name__ == '__main__':
    nb_episode = 1000

    agent1 = NashQLearner(
        actions=np.arange(2))
    agent2 = NashQLearner(
        actions=np.arange(2))

    game = MatrixGame()
    pi1_history = []
    pi2_history = []
    for episode in range(nb_episode):
        action1 = agent1.act()
        action2 = agent2.act()

        _, r1, r2 = game.step(action1, action2)

        agent1.observe(
            reward=r1,
            reward_o=r2,
            opponent_action=agent2.prev_action)
        agent2.observe(
            reward=r2,
            reward_o=r1,
            opponent_action=agent1.prev_action)

        pi1 = agent1.get_pi()
        pi2 = agent2.get_pi()
        pi1_history.append(pi1[0])
        pi2_history.append(pi2[0])

    plt.plot(np.arange(len(pi1_history)),
             pi1_history, label="agent1's pi(0)")
    plt.plot(np.arange(len(pi2_history)),
             pi2_history, label="agent2's pi(0)")
    plt.xlabel("episode")
    plt.ylabel("pi(0)")
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig("result.jpg")
    plt.show()
