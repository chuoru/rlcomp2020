import sys
import numpy as np
from GAME_SOCKET_DUMMY import GameSocket #in testing version, please use GameSocket instead of GAME_SOCKET_DUMMY
from MINER_STATE import State
from copy import deepcopy


TreeID = 1
TrapID = 2
SwampID = 3

SIGHT_X, SIGHT_Y = 2,2

class MinerEnv:
    def __init__(self, host, port):
        self.socket = GameSocket(host, port)
        self.state = State()
        self.gone_cell = []
        self.score_pre = self.state.score#Storing the last score for designing the reward function
        self.energy_pre = 50
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
        self.pre_map = self.get_map() #getting the old map for rewarding
        self.socket.send(action) #send action to server
        try:
            message = self.socket.receive() #receive new state from server
            self.state.update_state(message) #update to local state
        except Exception as e:
            import traceback
            traceback.print_exc()

    def change_sight(self, map, sight_x, sight_y):
        map_temp = deepcopy(map)
        for i in range(self.state.mapInfo.max_x + 1):
            for j in range(self.state.mapInfo.max_y + 1):
                if not (max(0, self.state.x - sight_x) <= i <= min(self.state.mapInfo.max_x, self.state.x + sight_x) and max(0,
                        self.state.y - sight_y) <= j <= min(
                        self.state.mapInfo.max_y, self.state.y + sight_y)) and map_temp[i, j] < 0:
                    map_temp[i, j] = 0
        return map_temp
    def get_map(self):
        view = np.zeros([self.state.mapInfo.max_x + 1, self.state.mapInfo.max_y + 1], dtype=int)
        for i in range(self.state.mapInfo.max_x + 1):
            for j in range(self.state.mapInfo.max_y + 1):
                if self.state.mapInfo.get_obstacle(i, j) == TreeID:  # Tree
                    view[i, j] = -TreeID
                if self.state.mapInfo.get_obstacle(i, j) == TrapID:  # Trap
                    view[i, j] = -TrapID
                if self.state.mapInfo.get_obstacle(i, j) == SwampID:  # Swamp
                    view[i, j] = -SwampID
                if self.state.mapInfo.gold_amount(i, j) > 0:
                    view[i, j] = self.state.mapInfo.gold_amount(i, j)
        return view

    # Functions are customized by client
    def get_state(self):
        # Building the map
        view = self.get_map()

        # view = self.change_sight(view, SIGHT_X, SIGHT_Y)

        DQNState = view.flatten().tolist() #Flattening the map matrix to a vector
        
        # Add position and energy of agent to the DQNState
        DQNState.append(self.state.x)
        DQNState.append(self.state.y)
        DQNState.append(self.state.energy)
        #Add position of bots 
        # for player in self.state.players:
        #     if player["playerId"] != self.state.id:
        #         DQNState.append(player["posx"])
        #         DQNState.append(player["posy"])
                
        #Convert the DQNState from list to array for training
        DQNState = np.array(DQNState)

        return DQNState

    def cell_had_had_gold(self, x, y):
        return self.pre_map[x, y] > 0

    def cell_has_gold(self, x, y):
        if (self.state.mapInfo.is_column_has_gold(x) and
                self.state.mapInfo.is_row_has_gold(y)):
            return True
        return False

    def toward_gold(self):
        direction = ((-1,0),(1,0),(0,-1),(0,1))
        action = self.state.lastAction
        if action < 4:
            in_r, in_c = direction[action]
        else:
            return 0
        x, y = self.state.x - in_r, self.state.y - in_c
        if (x, y, action) in self.gone_cell:
            return 0
        else: self.gone_cell.append((x, y, action))
        step_num = 0
        while True:
            step_num += 1
            try:
                if x + step_num * in_r < 0 or x + step_num * in_r > self.state.mapInfo.max_x or y + step_num * in_c < 0 or y + step_num * in_c > self.state.mapInfo.max_y:
                    break
                if self.cell_had_had_gold(x + step_num * in_r, y + step_num * in_c):
                    return self.pre_map[x + step_num * in_r,y
                                                    + step_num * in_c]/(step_num * step_num + 1)
            except IndexError:
                return 0
        return 0

    def leave_gold(self):
        direction = ((-1,0),(1,0),(0,-1),(0,1),(0,0),(0,0))
        action = self.state.lastAction
        if action <= 5:
            in_r, in_c = direction[action]
            return 2 * (self.cell_had_had_gold(self.state.x - in_r, self.state.y - in_c) and action < 5)
        else: return 0
# 　　　　　　　　　lastaction   examine
#                     |         |
#                     |         |
#                     v         v
#        --old--------o--------new-------->

    def get_reward(self):
        # Calculate reward
        reward = 0
        score_action = self.state.score - self.score_pre
        self.score_pre = self.state.score



        if score_action <= 0 and self.state.lastAction == 5:
            reward += -3 #fixed
        reward -= self.leave_gold() #fixed

        if self.energy_pre > 45 and self.state.lastAction == 4: reward += -2
        if self.energy_pre <= 10 and self.state.lastAction == 4: reward += 1
        self.energy_pre = self.state.energy # update the prior energy for next stage

        if score_action > 0:
            #If the DQN agent crafts golds, then it should obtain a positive reward (equal score_action)
            reward += score_action

        reward += self.toward_gold()  #fixed
        #If the DQN agent crashs into obstacels (Tree, Trap, Swamp), then it should be punished by a negative reward
        if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == TreeID:  # Tree
            reward -= abs(TreeID)*3
        if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == TrapID:  # Trap
            reward -= abs(TrapID)*2
        if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == SwampID:  # Swamp
            reward -= abs(SwampID)*2

        # If out of the map, then the DQN agent should be punished by a larger nagative reward.
        if self.state.status == State.STATUS_ELIMINATED_WENT_OUT_MAP:
            reward += -10
            
        #Run out of energy, then the DQN agent should be punished by a larger nagative reward.
        if self.state.status == State.STATUS_ELIMINATED_OUT_OF_ENERGY:
            reward += -10
        # print ("reward",reward)

        return reward

    def check_terminate(self):
        #Checking the status of the game
        #it indicates the game ends or is playing
        self.how_terminate = ['STATUS_PLAYING','STATUS_ELIMINATED_WENT_OUT_MAP','STATUS_ELIMINATED_OUT_OF_ENERGY',
    'STATUS_ELIMINATED_INVALID_ACTION', 'STATUS_STOP_EMPTY_GOLD', 'STATUS_STOP_END_STEP'][self.state.status]
        return self.state.status != State.STATUS_PLAYING
