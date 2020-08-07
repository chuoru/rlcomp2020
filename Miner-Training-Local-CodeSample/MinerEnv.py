import sys
import numpy as np
from GAME_SOCKET_DUMMY import GameSocket  # in testing version, please use GameSocket instead of GAME_SOCKET_DUMMY
from MINER_STATE import State
from copy import deepcopy

TreeID = 1
TrapID = 2
SwampID = 3
SIGHT_X = 1
SIGHT_Y = 1
SIGHT = 2

class MinerEnv:
    def __init__(self, host, port):
        self.socket = GameSocket(host, port)
        self.state = State()

        self.score_pre = self.state.score  # Storing the last score for designing the reward function
        self.energy_pre = self.state.energy

    def get_score(self):
        return self.score_pre

    def start(self):  # connect to server
        self.socket.connect()

    def end(self):  # disconnect server
        self.socket.close()

    def send_map_info(self, request):  # tell server which map to run
        self.socket.send(request)

    def reset(self):  # start new game
        try:
            message = self.socket.receive()  # receive game info from server
            self.state.init_state(message)  # init state
        except Exception as e:
            import traceback
            traceback.print_exc()

    def step(self, action):  # step process
        self.socket.send(action)  # send action to server
        try:
            message = self.socket.receive()  # receive new state from server
            self.state.update_state(message)  # update to local state
        except Exception as e:
            import traceback
            traceback.print_exc()

    def change_sight(self, map, sight_x, sight_y):
        map_temp = deepcopy(map)
        for i in range(self.state.mapInfo.max_x + 1):
            for j in range(self.state.mapInfo.max_y + 1):
                if not (max(0, self.state.x - sight_x) <= i <= min(self.state.mapInfo.max_x,
                                                                   self.state.x + sight_x) and max(0,
                                                                   self.state.y - sight_y) <= j <= min(
                                                                   self.state.mapInfo.max_y, self.state.y + sight_y)) \
                        and map_temp[i, j] < 0:
                    map_temp[i, j] = 0
        return map_temp

    def reduce_sight(self, visible_range):
        view = np.zeros([2*visible_range + 1, 2*visible_range + 1], dtype=int)
        for x in range(-visible_range, visible_range + 1):
            for y in range(-visible_range, visible_range + 1):
                if (self.state.x - x >= 0) and (self.state.y - y >= 0) and\
                        (self.state.x + x <= self.state.mapInfo.max_x) and (self.state.y + y <= self.state.mapInfo.max_y):
                    view[visible_range - x, visible_range - y] = self.get_content(self.state.x - x, self.state.y - y)
                else:
                    view[visible_range - x, visible_range - y] = -100
        return view

    def get_content(self, i, j):
        content = 0
        if self.state.mapInfo.get_obstacle(i, j) == TreeID:  # Tree
            content = -TreeID
        if self.state.mapInfo.get_obstacle(i, j) == TrapID:  # Trap
            content = -TrapID
        if self.state.mapInfo.get_obstacle(i, j) == SwampID:  # Swamp
            content = -SwampID
        if self.state.mapInfo.gold_amount(i, j) > 0:
            content = self.state.mapInfo.gold_amount(i, j)
        return content

    # Functions are customized by client
    def get_state(self):
        # Building the map
        view = np.zeros([self.state.mapInfo.max_x + 1, self.state.mapInfo.max_y + 1, 4], dtype=int)
        for i in range(self.state.mapInfo.max_x + 1):
            for j in range(self.state.mapInfo.max_y + 1):
                if self.state.mapInfo.get_obstacle(i, j) == TreeID:  # Tree
                    view[i, j, 0] = -TreeID
                if self.state.mapInfo.get_obstacle(i, j) == TrapID:  # Trap
                    view[i, j, 0] = -TrapID
                if self.state.mapInfo.get_obstacle(i, j) == SwampID: # Swamp
                    view[i, j, 0] = -SwampID
                if self.state.mapInfo.gold_amount(i, j) > 0:
                    view[i, j, 1] = self.state.mapInfo.gold_amount(i, j)

        view[self.state.x, self.state.y, 2] = 1
        # Add position of bots
        for player in self.state.players:
            if player["playerId"] != self.state.id:
                view[player["posx"], player["posy"], 2] = -1

        # Add position and energy of agent to the DQNState
        view[self.state.x, self.state.y, 3] = self.state.energy

        # Convert the DQNState from list to array for training
        DQNState = np.array(view)

        return DQNState

    def get_reward(self):
        # Calculate reward
        reward = 0
        score_action = self.state.score - self.score_pre
        self.score_pre = self.state.score
        if score_action > 0:
            reward += 2 * score_action

        if self.state.mapInfo.gold_amount(self.state.x, self.state.y) >= 100:
            # If the DQN agent crafts golds, then it should obtain a positive reward (equal score_action)
            reward += 5
            if (self.state.lastAction is 5) and (self.state.energy > 15):
                reward += 20

        if (self.state.mapInfo.is_row_has_gold(self.state.x)) and ((self.state.lastAction is 0) or (self.state.lastAction is 1)):
            reward += 1

        if (self.state.mapInfo.is_column_has_gold(self.state.x)) and ((self.state.lastAction is 2) or (self.state.lastAction is 3)):
            reward += 1

        if (self.state.energy is 50) and (self.state.lastAction is 4):
            reward += -10

        # if (self.state.energy <= 10) and (self.state.lastAction is 4):
        #     reward += 5

        # If the DQN agent crashs into obstacels (Tree, Trap, Swamp), then it should be punished by a negative reward
        if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == TreeID:  # Tree
            reward -= TreeID
        if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == TrapID:  # Trap
            reward -= TrapID
        if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == SwampID:  # Swamp
            reward -= SwampID

        # If out of the map, then the DQN agent should be punished by a larger nagative reward.
        if self.state.status == State.STATUS_ELIMINATED_WENT_OUT_MAP:
            reward += -20

        # Run out of energy, then the DQN agent should be punished by a larger nagative reward.
        if self.state.status == State.STATUS_ELIMINATED_OUT_OF_ENERGY:
            reward += -10
        # print ("reward",reward)
        return reward

    def check_terminate(self):
        # Checking the status of the game
        # it indicates the game ends or is playing
        return self.state.status != State.STATUS_PLAYING
