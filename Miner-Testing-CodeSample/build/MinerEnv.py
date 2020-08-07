from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)

import sys
import numpy as np
from GAME_SOCKET import GameSocket #in testing version, please use GameSocket instead of GameSocketDummy
from MINER_STATE import State
from copy import deepcopy

SIGHT_X = 1
SIGHT_Y = 1
TreeID = 1
TrapID = 2
SwampID = 3
SIGHT = 2

class MinerEnv:
    def __init__(self, host, port):
        self.socket = GameSocket(host, port)
        self.state = State()
        
        self.score_pre = self.state.score#Storing the last score for designing the reward function
        
    def start(self): #connect to server
        self.socket.connect()

    def end(self): #disconnect server
        self.socket.close()

    def send_map_info(self, request):#tell server which map to run
        self.socket.send(request)

    def reset(self): #start new game
        try:
            message = self.socket.receive() #receive game info from server
            self.state.init_state(message) #init state
        except Exception as e:
            import traceback
            traceback.print_exc()

    def step(self, action): #step process
        self.socket.send(action) #send action to server
        try:
            message = self.socket.receive() #receive new state from server
            #print("New state: ", message)
            self.state.update_state(message) #update to local state
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
                if (self.state.x - x >= 0) and (self.state.y - y >= 0) and \
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
        # view = np.zeros([self.state.mapInfo.max_x + 1, self.state.mapInfo.max_y + 1], dtype=int)
        # for i in range(self.state.mapInfo.max_x + 1):
        #     for j in range(self.state.mapInfo.max_y + 1):
        #         if self.state.mapInfo.get_obstacle(i, j) == TreeID:  # Tree
        #             view[i, j] = -TreeID
        #         if self.state.mapInfo.get_obstacle(i, j) == TrapID:  # Trap
        #             view[i, j] = -TrapID
        #         if self.state.mapInfo.get_obstacle(i, j) == SwampID: # Swamp
        #             view[i, j] = -SwampID
        #         if self.state.mapInfo.gold_amount(i, j) > 0:
        #             view[i, j] = self.state.mapInfo.gold_amount(i, j)
        #
        # reduce_view = self.change_sight(view, SIGHT_X, SIGHT_Y)

        reduce_view = self.reduce_sight(SIGHT)

        DQNState = reduce_view.flatten().tolist() #Flattening the map matrix to a vector
        # Add position and energy of agent to the DQNState
        DQNState.append(self.state.x)
        DQNState.append(self.state.y)
        DQNState.append(self.state.energy)
        # Add position of bots
        # for player in self.state.players:
        #     if player["playerId"] != self.state.id:
        #         DQNState.append(player["posx"])
        #         DQNState.append(player["posy"])

        # Convert the DQNState from list to array for training
        DQNState = np.array(DQNState)

        return DQNState

    def check_terminate(self):
        return self.state.status != State.STATUS_PLAYING
