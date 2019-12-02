import numpy as np
import nashpy


class NashQLearner():
    def __init__(self,
                 alpha=0.1,
                 policy=None,
                 gamma=0.99,
                 ini_state="nonstate",
                 actions=None):

        self.alpha = alpha
        self.gamma = gamma
        self.policy = policy
        self.actions = actions
        self.state = ini_state

        # q values (my and opponent)
        self.q, self.q_o = {}, {}
        self.q[ini_state] = {}
        self.q_o[ini_state] = {}

        # nash q value
        self.nashq = {}
        self.nashq[ini_state] = 0

        # pi (my and opponent)
        self.pi, self.pi_o = {}, {}
        self.pi[ini_state] = np.repeat(1.0/len(self.actions), len(self.actions))
        self.pi_o[ini_state] = np.repeat(1.0/len(self.actions), len(self.actions))

        self.previous_action = None
        self.reward_history = []
        self.pi_history = []

    def act(self, training=True):
        if training:
            action_id = self.policy.select_action(self.pi[self.state])
            action = self.actions[action_id]
            self.previous_action = action
        else:
            action_id = self.policy.select_greedy_action(self.pi)
            action = self.actions[action_id]

        return action

    def observe(self, state="nonstate", reward=None, reward_o=None, opponent_action=None, is_learn=True):
        """
            observe next state and learn 
        """
        if is_learn:
            self.check_new_state(state) # if the state is new state, extend q table
            self.learn(state, reward, reward_o, opponent_action)

    def learn(self, state, reward, reward_o, opponent_action):
        self.reward_history.append(reward)
        self.q[state][(self.previous_action, opponent_action)] = self.compute_q(state, reward, opponent_action, self.q)
        self.q_o[state][(self.previous_action, opponent_action)] = self.compute_q(state, reward_o, opponent_action, self.q_o)

        self.pi[state], self.pi_o[state] = self.compute_pi(state)
        self.nashq[state] = self.compute_nashq(state)

        self.pi_history.append(self.pi[state][0])

    def compute_q(self, state, reward, opponent_action, q):
        if (self.previous_action, opponent_action) not in q[state].keys():
            q[state][(self.previous_action, opponent_action)] = 0.0
        q_old = q[state][(self.previous_action, opponent_action)]
        updated_q = q_old + (self.alpha * (reward + self.gamma*self.nashq[state] - q_old))

        return updated_q

    def compute_nashq(self, state):
        """
            compute nash q value 
        """
        nashq = 0
        for action1 in self.actions:
            for action2 in self.actions:
                nashq += self.pi[state][action1]*self.pi_o[state][action2] * \
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
                row_q_2.append(self.q_o[state][joint_action])
            q_1.append(row_q_1)
            q_2.append(row_q_2)

        game = nashpy.Game(q_1, q_2)
        equilibria = game.support_enumeration()
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
        for action1 in self.actions:
            for action2 in self.actions:
                if state not in self.pi.keys():
                    self.pi[state] = np.repeat(
                        1.0/len(self.actions), len(self.actions))
                    self.v[state] = np.random.random()
                if (action1, action2) not in self.q[state].keys():
                    self.q[state][(action1, action2)] = np.random.random()
                    self.q_o[state][(action1, action2)] = np.random.random()
