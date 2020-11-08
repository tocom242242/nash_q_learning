import numpy as np
import matplotlib.pyplot as plt
from nash_q_learner import NashQLearner
from policy import EpsGreedyQPolicy
from grid_env import GridEnv

if __name__ == '__main__':
    nb_episode = 500
    max_steps = 100
    actions = np.arange(4)

    game = GridEnv(nb_agents=2)
    ini_pos = game.create_observations()
    agent1 = NashQLearner(
        id=0,
        alpha=0.1,
        policy=EpsGreedyQPolicy(
            epsilon=1.,
            decay_rate=0.999,
            min_epsilon=0.01),
        ini_state=ini_pos[0],
        actions=actions)
    agent2 = NashQLearner(
        id=1,
        alpha=0.1,
        policy=EpsGreedyQPolicy(
            epsilon=1.,
            decay_rate=0.999,
            min_epsilon=0.01),
        ini_state=ini_pos[1],
        actions=actions)

    game.print_map()
    game.print_agent_pos()
    step_history = []
    reward_history = {"0": [], "1": []}
    for episode in range(nb_episode):
        # print("======================================")
        for step in range(max_steps):
            actions = {}
            for agent in [agent1, agent2]:
                action = agent.act()
                actions[agent.id] = action
            observations, rewards, is_terminal = game.step(actions)

            agent1.observe(
                state=observations[0],
                reward=rewards[0],
                reward_o=rewards[1],
                opponent_action=agent2.prev_action,
            )
            agent2.observe(
                state=observations[1],
                reward=rewards[1],
                reward_o=rewards[0],
                opponent_action=agent1.prev_action,
            )

            for agent in [agent1, agent2]:
                agent.policy.update_epsilon()

            if is_terminal:
                break

        for agent in [agent1, agent2]:
            agent.update_pi()
        # print(f"{episode}:{step}")
        step_history.append(step)
        reward_history["0"].append(np.mean(agent1.reward_history))
        reward_history["1"].append(np.mean(agent2.reward_history))
        # game.print_map()
        # print(game.agents_pos)

        observations = game.reset()
        agent1.reset()
        agent1.observe(state=observations[0], learning=False)
        agent2.reset()
        agent2.observe(state=observations[1], learning=False)

    print(step_history)
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(len(step_history)), step_history, label="step")
    plt.legend()
    plt.subplot(3, 1, 2)
    reward_history["0"] = np.array(reward_history["0"])
    reward_history["1"] = np.array(reward_history["1"])
    plt.plot(np.arange(len(reward_history["0"])),
             reward_history["0"], label="reward_history0")
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(np.arange(len(reward_history["1"])),
             reward_history["1"], label="reward_history1")
    # ave = (reward_history["0"] + reward_history["1"]) / 2
    # plt.plot(np.arange(len(ave)),
    #          ave,
    #          label="average_reward_history")
    plt.legend()
    plt.savefig("test.png")
    plt.show()
    # plt.plot(np.arange(len(agent1.pi_history)), agent1.pi_history, label="agent1's pi(0)")
    # plt.plot(np.arange(len(agent2.pi_history)), agent2.pi_history, label="agent2's pi(0)")
    # plt.xlabel("episode")
    # plt.ylabel("pi(0)")
    # plt.legend()
    # plt.savefig("result.jpg")
    # plt.show()
