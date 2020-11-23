import numpy as np
import matplotlib.pyplot as plt
from nash_q_learner import NashQLearner
from grid_game import GridGame


def run_episode(learning, max_steps, agents, game, is_plot=False):
    for step in range(max_steps):
        actions = {}
        for agent in agents:
            action = agent.act(training=learning)
            actions[agent.id] = action
        observations, rewards, is_terminal = game.step(actions)

        agents[0].observe(
            state=observations[0],
            reward=rewards[0],
            reward_o=rewards[1],
            opponent_action=agents[1].prev_action,
            learning=learning
        )

        agents[1].observe(
            state=observations[1],
            reward=rewards[1],
            reward_o=rewards[0],
            opponent_action=agents[0].prev_action,
            learning=learning
        )

        if is_plot:
            game.print_map()

        if is_terminal:
            break

    average_rewards = []
    if is_plot:
        print(agents[0].reward_history)
        print(agents[1].reward_history)
        print(agents[0].action_history)
        print(agents[1].action_history)

    average_rewards.append(np.mean(agents[0].reward_history))
    average_rewards.append(np.mean(agents[1].reward_history))

    return step, average_rewards


if __name__ == '__main__':
    nb_episode = 30000
    max_steps = 10000
    actions = np.arange(4)

    game = GridGame(nb_agents=2)
    ini_pos = game.create_observations()

    agent1 = NashQLearner(
        id=0,
        epsilon=0.02,
        ini_state=ini_pos[0],
        actions=actions)
    agent2 = NashQLearner(
        id=1,
        epsilon=0.02,
        ini_state=ini_pos[1],
        actions=actions)

    step_history = []
    reward_history = {"0": [], "1": []}
    for episode in range(nb_episode):
        observations = game.reset()
        agent1.reset(state=observations[0])
        agent2.reset(state=observations[1])
        step, rewards = run_episode(
            learning=True, max_steps=max_steps, agents=[
                agent1, agent2], game=game)

        if episode % 500 == 0:
            # test
            observations = game.reset(pos_list=((0, 1), (2, 1)))
            agent1.reset(state=observations[0])
            agent2.reset(state=observations[1])

            is_plot = False
            step, rewards = run_episode(
                learning=False, max_steps=max_steps, agents=[
                    agent1, agent2], game=game, is_plot=is_plot)

            step_history.append(step)
            reward_history["0"].append(rewards[0])
            reward_history["1"].append(rewards[1])
            print("------------------------------------")
            print(f"{episode},step:{step}, a0:{rewards[0]},a1:{rewards[1]}")
            print("------------------------------------")

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
    plt.ylim(-50, 30)
    plt.subplot(3, 1, 3)
    plt.plot(np.arange(len(reward_history["1"])),
             reward_history["1"], label="reward_history1")
    plt.ylim(-50, 30)
    plt.legend()
    plt.savefig("result.png")
    plt.show()
