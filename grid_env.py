import numpy as np
import copy

F = {
    "N": 0,  # 通常
    "G": 1,  # 壁
    "W": 2,  # 壁
    "A": 3,  # 壁
}

A = {
    "UP": 0,
    "DOWN": 1,
    "LEFT": 2,
    "RIGHT": 3,
}


class GridEnv:

    def __init__(self, nb_agents):

        self.map = [[0, F["G"], 0],
                    [0, 0, 0],
                    [0, 0, 0]]

        # self.range_obs = range_obs
        self.range_obs = 1
        self.map = self.shape_map(self.range_obs, self.map)
        self.inimap = copy.deepcopy(self.map)

        self.nb_agents = nb_agents
        self.agents_pos = {}

        for aid in range(self.nb_agents):
            pos = self._get_random_pos(self.agents_pos.values(), F["A"])
            self._set_pos(x=pos[1], y=pos[0], otype=F["A"], aid=aid)
            self.agents_pos[aid] = pos

        # self._set_pos(x=1, y=3, otype=F["A"], aid=0)
        # self._set_pos(x=3, y=3, otype=F["A"], aid=1)

        self.ini_agent_pos = copy.deepcopy(self.agents_pos)

        self.is_goals = [False for _ in range(self.nb_agents)]

        self.inimap = copy.deepcopy(self.map)

    def _get_random_pos(self, poss=[], otype=None):
        """
            被らないposデータの生成
        """

        x = np.random.randint(
            self.range_obs, len(
                self.map[0]) - self.range_obs)
        y = np.random.randint(self.range_obs, len(self.map) - self.range_obs)

        while ((x, y) in poss or self.map[y][x]
               == F["W"] or self.map[y][x] == F["G"]):
            x = np.random.randint(
                self.range_obs, len(
                    self.map[0]) - self.range_obs)
            y = np.random.randint(
                self.range_obs, len(
                    self.map) - self.range_obs)

        # self.map[y][x] = otype
        return x, y

    def shape_map(self, range_obs, _gridmap):
        gridmap = _gridmap
        for _ in range(range_obs):
            _r = np.full((1, len(gridmap[0])), 2)
            gridmap = np.concatenate((gridmap, _r), axis=0)
            gridmap = np.concatenate((_r, gridmap), axis=0)
            _c = np.full((1, len(gridmap)), 2).T
            gridmap = np.concatenate((gridmap, _c), axis=1)
            gridmap = np.concatenate((_c, gridmap), axis=1)

        return gridmap

    def step(self, actions):
        """
            全エージェントの行動の実行
            状態, 報酬、ゴールしたかを返却
        """
        next_positions = {}
        prev_positions = {}
        for aid, action in actions.items():
            x, y = copy.deepcopy(self.agents_pos[aid])
            prev_positions[aid] = (x, y)
            if self.map[y][x] != F["G"]:
                self.map[y][x] = F["N"]
            to_x, to_y = self.move(x, y, action)
            next_positions[aid] = (to_x, to_y)

        # ゴール判定
        for aid, pos in next_positions.items():
            if self.map[pos[1]][pos[0]] == F["G"]:
                self.is_goals[aid] = True
            else:
                self.is_goals[aid] = False
        if all(self.is_goals):
            print(self.is_goals)
        # 衝突判定
        is_collisions = {}
        for aid, pos in next_positions.items():
            is_collision = self.check_collision(
                pos, aid, next_positions) or self._is_walls(
                pos[0], pos[1])
            is_collisions[aid] = is_collision

        # 報酬、移動
        rewards = {}
        is_terminal = False
        for aid, action in actions.items():
            if self.is_goals[aid]:
                rewards[aid] = 100
                to_x, to_y = next_positions[aid]
                is_terminal = True
            elif is_collisions[aid]:
                rewards[aid] = -1
                to_x, to_y = prev_positions[aid]
                self.map[to_y][to_x] = F["A"]
            else:
                rewards[aid] = 0
                to_x, to_y = next_positions[aid]
                self.map[to_y][to_x] = F["A"]

            self.agents_pos[aid] = (to_x, to_y)

        # 観測
        observations = self.create_observations()
        return observations, rewards, is_terminal

    def check_collision(self, pos, aid, positions):
        for aid2, pos2 in positions.items():
            if pos == pos2 and aid != aid2:
                if self.is_goals[aid2] is False and self.is_goals[aid] is False:
                    return True

        return False

    def create_observations(self):
        observations = {}
        for aid in range(self.nb_agents):
            obs = self.create_observation()
            observations[aid] = obs
        return observations

    def create_observation(self):
        """
            観測情報の生成
        """
        observation = []
        for aid in range(self.nb_agents):
            observation.append(self.agents_pos[aid])

        observation = tuple(observation)
        return observation

    def _set_pos(self, x, y, otype, aid):
        self.map[y][x] = otype
        self.agents_pos[aid] = x, y
        return x, y

    def move(self, x, y, action):
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

        return to_x, to_y

    def _is_walls(self, x, y):
        """
            実行可能な行動かどうかの判定
        """
        to_x, to_y = x, y

        if len(self.map) <= to_y or 0 > to_y:
            return True
        elif len(self.map[0]) <= to_x or 0 > to_x:
            return True
        elif self.map[to_y][to_x] == F["W"]:
            return True

        return False

    def reset(self):
        self.map = copy.deepcopy(self.inimap)
        self.agents_pos = {}
        for aid in range(self.nb_agents):
            pos = self._get_random_pos(self.agents_pos.values(), F["A"])
            self._set_pos(x=pos[1], y=pos[0], otype=F["A"], aid=aid)
            self.agents_pos[aid] = pos
        # self.agents_pos = copy.deepcopy(self.ini_agent_pos)
        self.is_goals = [False for _ in range(self.nb_agents)]

        observations = self.create_observations()
        return observations

    def get_agents_ini_pos(self):
        return self.agents_ini_pos

    def get_agents_pos(self):
        return self.agents_pos

    def print_agent_pos(self):
        print(self.agents_pos)

    def print_map(self):
        print("------------------------------------------------")
        print(np.array(self.map))
