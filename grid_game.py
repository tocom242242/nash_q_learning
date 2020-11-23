import numpy as np
import copy

F = {
    "N": 0,  # Normal
    "G1": 1,  # Goal1
    "G2": 2,  # Goal2
    "A": 3,  # Agent1
    "B": 4,  # Agent2
}

A = {
    "UP": 0,
    "DOWN": 1,
    "LEFT": 2,
    "RIGHT": 3,
}


class GridGame:

    def __init__(self, nb_agents):
        self.map = [[F["G2"], F["N"], F["G1"]],
                    [F["N"], F["N"], F["N"]]]

        self.inimap = copy.deepcopy(self.map)

        self.nb_agents = nb_agents
        self.agents_pos = {}

        for aid in range(self.nb_agents):
            agent_mark = self._get_agent_mark(aid)
            pos = self._get_random_pos(self.agents_pos.values(), agent_mark)
            self._set_pos(x=pos[0], y=pos[1], agent_mark=agent_mark, aid=aid)
            self.agents_pos[aid] = pos

        self.ini_agent_pos = copy.deepcopy(self.agents_pos)
        self.is_goals = [False for _ in range(self.nb_agents)]

    def _get_agent_mark(self, aid):
        if aid == 0:
            agent_mark = F["A"]
        elif aid == 1:
            agent_mark = F["B"]

        return agent_mark

    def _get_random_pos(self, poss=[], agent_mark=None):
        """
            generate agent's position randomly
        """

        x = np.random.randint(0, len(self.map[0]))
        y = np.random.randint(0, len(self.map))

        while ((x, y) in poss or self.map[y][x] in [F["G1"], F["G2"]]):
            x = np.random.randint(0, len(self.map[0]))
            y = np.random.randint(0, len(self.map))
        return x, y

    def step(self, actions):
        next_positions = {}
        prev_positions = {}
        for aid, action in actions.items():
            x, y = copy.deepcopy(self.agents_pos[aid])
            prev_positions[aid] = (x, y)
            if self.map[y][x] not in [F["G1"], F["G2"]]:
                self.map[y][x] = F["N"]
            to_x, to_y = self._move(x, y, action)
            next_positions[aid] = (to_x, to_y)

        # check agent collides with wall
        is_walls = {}
        for aid, pos in next_positions.items():
            is_wall = self._is_walls(pos[0], pos[1])
            is_walls[aid] = is_wall
            if is_wall:
                next_positions[aid] = prev_positions[aid]

        # check each agent arrives their goal
        for aid, pos in next_positions.items():
            if self.is_goals[aid] is False:
                if aid == 0 and self.map[pos[1]][pos[0]] == F["G1"]:
                    self.is_goals[aid] = True
                elif aid == 1 and self.map[pos[1]][pos[0]] == F["G2"]:
                    self.is_goals[aid] = True
                else:
                    self.is_goals[aid] = False

        # check whether agent collides with other agent
        is_collisions = {}
        for aid, pos in next_positions.items():
            is_collision = self._check_collision(pos, aid, next_positions)
            is_collisions[aid] = is_collision

        rewards = {}
        is_terminal = False
        for aid, action in actions.items():
            if self.is_goals[aid]:
                rewards[aid] = 100
                to_x, to_y = next_positions[aid]
            elif is_collisions[aid]:
                rewards[aid] = -10
                to_x, to_y = prev_positions[aid]
                if self.map[to_y][to_x] not in [F["G1"], F["G2"]]:
                    agent_mark = self._get_agent_mark(aid)
                    self.map[to_y][to_x] = agent_mark
            elif is_walls[aid]:
                rewards[aid] = -10
                to_x, to_y = prev_positions[aid]
                if self.map[to_y][to_x] not in [F["G1"], F["G2"]]:
                    agent_mark = self._get_agent_mark(aid)
                    self.map[to_y][to_x] = agent_mark
            else:
                rewards[aid] = -1
                to_x, to_y = next_positions[aid]
                if self.map[to_y][to_x] not in [F["G1"], F["G2"]]:
                    agent_mark = self._get_agent_mark(aid)
                    self.map[to_y][to_x] = agent_mark

            self.agents_pos[aid] = (to_x, to_y)

        if all(self.is_goals):
            is_terminal = True

        observations = self.create_observations()
        return observations, rewards, is_terminal

    def _check_collision(self, pos, aid, positions):
        for aid2, pos2 in positions.items():
            if pos == pos2 and aid != aid2:
                if self.is_goals[aid2] is False and self.is_goals[aid] is False:
                    return True

        return False

    def create_observations(self):
        observation = []
        for aid in range(self.nb_agents):
            observation.append(self.agents_pos[aid])
        observation = tuple(observation)

        observations = {}
        observations[0] = observation
        observations[1] = observation
        return observations

    def _set_pos(self, x, y, agent_mark, aid):
        self.map[y][x] = agent_mark
        self.agents_pos[aid] = x, y
        return x, y

    def _move(self, x, y, action):
        to_x = copy.deepcopy(x)
        to_y = copy.deepcopy(y)
        if action == A["UP"]:
            to_y += -1
        elif action == A["DOWN"]:
            to_y += 1
        elif action == A["LEFT"]:
            to_x += -1
        elif action == A["RIGHT"]:
            to_x += 1
        else:
            import IPython
            IPython.embed(header="action")

        return to_x, to_y

    def _is_walls(self, x, y):
        if len(self.map) <= y or 0 > y:
            return True
        elif len(self.map[0]) <= x or 0 > x:
            return True

        return False

    def reset(self, pos_list=None):
        self.map = copy.deepcopy(self.inimap)
        self.agents_pos = {}

        if pos_list is None:
            pos_list = []
            for aid in range(self.nb_agents):
                agent_mark = self._get_agent_mark(aid)
                pos = self._get_random_pos(
                    self.agents_pos.values(), agent_mark)
                pos_list.append(pos)

        for aid in range(self.nb_agents):
            pos = pos_list[aid]
            agent_mark = self._get_agent_mark(aid)
            self._set_pos(x=pos[0], y=pos[1], agent_mark=agent_mark, aid=aid)
            self.agents_pos[aid] = pos
        self.is_goals = [False for _ in range(self.nb_agents)]

        observations = self.create_observations()
        return observations

    def print_map(self):
        """
            for debug
        """
        print("------------------------------------------------")
        print(np.array(self.map))
