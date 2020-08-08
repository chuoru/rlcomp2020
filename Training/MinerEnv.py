import sys
import numpy as np
from GAME_SOCKET_DUMMY import GameSocket #in testing version, please use GameSocket instead of GAME_SOCKET_DUMMY
from MINER_STATE import State
from copy import deepcopy

import pygame as pg
from pygame.locals import *
import sys
import time


TILE_SIZE = 40
MAP_WIDTH, MAP_HEIGHT = 5, 5
X_MARGIN, Y_MARGIN = TILE_SIZE, TILE_SIZE
WIN_WIDTH, WIN_HEIGHT = X_MARGIN + TILE_SIZE * MAP_WIDTH + 100, Y_MARGIN + TILE_SIZE * MAP_HEIGHT + 50


color = {
'GOLD' : (255, 255, 0),
'SWAMP' : (120, 148, 132),
'TRAP' : (255, 141, 56),
'MINER' : (219, 119, 141),
'TREE' : (59, 198, 182), # (0,95,95)
'BACKGROUND' : (44, 62, 80),
'BOT' : (192, 192, 192),
'FONT' : (46, 134, 193),}

pg.init()
pg.display.set_caption('DQN Miner Visualize')
Surf = pg.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
Surf.fill(color['BACKGROUND'])
fontObj = pg.font.Font('freesansbold.ttf', 15)

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

    def visualize(self):
        Surf.fill(color['BACKGROUND'])
        pg.draw.line(Surf, (255, 0, 0), (X_MARGIN + TILE_SIZE * (self.state.mapInfo.max_x + 1), Y_MARGIN), (X_MARGIN + TILE_SIZE * (self.state.mapInfo.max_x + 1), Y_MARGIN + TILE_SIZE * (self.state.mapInfo.max_y + 1)), 4)
        pg.draw.line(Surf, (255, 0, 0), (X_MARGIN, Y_MARGIN + TILE_SIZE * (self.state.mapInfo.max_y + 1)), (X_MARGIN + TILE_SIZE * (self.state.mapInfo.max_x + 1), Y_MARGIN + TILE_SIZE * (self.state.mapInfo.max_y + 1)), 4)
        pg.draw.line(Surf, (255, 0, 0), (X_MARGIN + TILE_SIZE * (self.state.mapInfo.max_x + 1), Y_MARGIN), (X_MARGIN, Y_MARGIN), 4)
        pg.draw.line(Surf, (255, 0, 0), (X_MARGIN, Y_MARGIN + TILE_SIZE * (self.state.mapInfo.max_y + 1)), (X_MARGIN, Y_MARGIN), 4)

        self.map = self.get_map()

        #draw the trap: orange, tree : green, swamp : dark green, bots: Grey and player : pynk
        for i in range(len(self.map)):
            for j in range(len(self.map[0])):
                if self.map[i, j] == -TreeID:
                    pg.draw.rect(Surf, color['TREE'], (X_MARGIN + i * TILE_SIZE, Y_MARGIN + j * TILE_SIZE, TILE_SIZE, TILE_SIZE))
                if self.map[i, j] == -TrapID:
                    pg.draw.rect(Surf, color['TRAP'], (X_MARGIN + i * TILE_SIZE, Y_MARGIN + j * TILE_SIZE, TILE_SIZE, TILE_SIZE))
                if self.map[i, j] == -SwampID:
                    pg.draw.rect(Surf, color['SWAMP'], (X_MARGIN + i * TILE_SIZE, Y_MARGIN + j * TILE_SIZE, TILE_SIZE, TILE_SIZE))
                if self.state.mapInfo.gold_amount(i, j) > 0:
                    pg.draw.rect(Surf, color['GOLD'], (X_MARGIN + i * TILE_SIZE, Y_MARGIN + j * TILE_SIZE, TILE_SIZE, TILE_SIZE))
        for bot in self.socket.bots:
            i, j = bot.info.posx, bot.info.posy
            pg.draw.rect(Surf, color['BOT'], (X_MARGIN + i * TILE_SIZE, Y_MARGIN + j * TILE_SIZE, TILE_SIZE, TILE_SIZE))
        pg.draw.rect(Surf, color['MINER'], (X_MARGIN + self.state.x * TILE_SIZE, Y_MARGIN + self.state.y * TILE_SIZE, TILE_SIZE, TILE_SIZE))

        #energy of the player
        textSurfaceEnergy = fontObj.render('''Energy: %s '''%(self.state.energy), True, color['FONT'], color['BACKGROUND'])
        textRectEnergy = textSurfaceEnergy.get_rect()
        textRectEnergy.right, textRectEnergy.top = WIN_WIDTH - 10, 10
        Surf.blit(textSurfaceEnergy, textRectEnergy)

    def update_map_visualize(self):
        for event in pg.event.get():
            if event.type == QUIT:
                pg.quit()
                sys.exit()
        pg.display.update()
        #time.sleep(0.3)

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
