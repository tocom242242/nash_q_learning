import numpy as np
import nashpy


class NashQLearner():
    def __init__(self,
                 id=None,
                 alpha=0.1,
                 policy=None,
                 gamma=0.99,
                 ini_state="nonstate",
                 actions=None):

        self.id = id
        self.alpha = alpha
        self.gamma = gamma
        self.policy = policy
        self.actions = actions
        self.state = ini_state
        self.prev_action = 0
        self.reward_history = []
        self.pi_history = []

        # q values (my and opponent)
        self.q, self.q_o = {}, {}
        # nash q value
        self.nashq = {}
        # pi (my and opponent)
        self.pi, self.pi_o = {}, {}
        self.n = {}  # counter

        self.check_new_state(ini_state)

    def act(self, training=True):
        if training:
            action_id = self.policy.select_action(self.pi[self.state])
            action = self.actions[action_id]
            self.prev_action = action
        else:
            action_id = self.policy.select_greedy_action(self.pi)
            action = self.actions[action_id]

        return action

    def observe(
            self,
            state="nonstate",
            reward=None,
            reward_o=None,
            opponent_action=0,
            learning=True):
        """
            observe next state and learn
        """
        self.check_new_state(state)
        self.n[(state, self.prev_action, opponent_action)] += 1
        if learning:
            self.learn(state, reward, reward_o, opponent_action)

    def learn(self, state, reward, reward_o, opponent_action):
        self.state = state
        self.reward_history.append(reward)
        self.q[state][(self.prev_action, opponent_action)] = self.compute_q(
            state, reward, self.prev_action, opponent_action, self.q)
        self.q_o[state][(opponent_action, self.prev_action)] = self.compute_q(
            state, reward_o, opponent_action, self.prev_action, self.q_o)

        self.nashq[state] = self.compute_nashq(state)

        self.pi_history.append(self.pi[state][0])

    def update_pi(self):
        self.pi[self.state], self.pi_o[self.state] = self.compute_pi(
            self.state)

    def compute_q(self, state, reward, own_action, opponent_action, q):
        if (own_action, opponent_action) not in q[state].keys():
            q[state][(own_action, opponent_action)] = 0.0
        q_old = q[state][(own_action, opponent_action)]
        self.alpha = 1 / (1 + self.n[(state, own_action, opponent_action)])
        updated_q = q_old + \
            (self.alpha * (reward + self.gamma * self.nashq[state] - q_old))

        return updated_q

    def compute_nashq(self, state):
        """
            compute nash q value
        """
        nashq = 0
        for action1 in self.actions:
            for action2 in self.actions:
                nashq += self.pi[state][action1] * self.pi_o[state][action2] * \
                    self.q[state][(action1, action2)]

        return nashq

    def compute_pi(self, state):
        """
            compute pi (nash)
        """
        q_1, q_2 = [], []
        for action1 in self.actions:
            row_q_1, row_q_2 = [], []
            for action2 in self.actions:
                joint_action = (action1, action2)
                row_q_1.append(self.q[state][joint_action])
                joint_action2 = (action1, action2)
                row_q_2.append(self.q_o[state][joint_action2])
            q_1.append(row_q_1)
            q_2.append(row_q_2)

        game = nashpy.Game(q_1, q_2)
        equilibria = game.lemke_howson_enumeration()
        pi = []
        for eq in equilibria:
            pi.append(eq)

        return pi[0][0], pi[0][1]

    def check_new_state(self, state):
        """
            if the state is new state, extend q table
        """

        if state not in self.q.keys():
            self.q[state] = {}
            self.q_o[state] = {}
            self.pi[state] = np.repeat(
                1.0 / len(self.actions), len(self.actions))
            self.pi_o[state] = np.repeat(
                1.0 / len(self.actions), len(self.actions))
            self.nashq[state] = np.random.random()
            for action1 in self.actions:
                for action2 in self.actions:
                    self.q[state][(action1, action2)] = np.random.random()
                    self.q_o[state][(action1, action2)] = np.random.random()
                    self.n[(state, action1, action2)] = 0

    def reset(self):
        self.reward_history = []
