from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)

import sys
from keras.models import model_from_json
from MinerEnv import MinerEnv
import numpy as np

def print_map(map, action_ls, x = 0, y = 0):
    action_to_dir = [(-1,0), (1,0), (0,-1), (0, 1)]
    action_to_dir2 = '<>^v'
    refer = [action_to_dir2[index] for index in action_ls if index < 4]
    positions = [(0,0)]
    for i in action_ls:
        if i < 4:
            positions.append((positions[-1][0] + action_to_dir[i][0], positions[-1][1] + action_to_dir[i][1]))
    max_x = len(map[0])
    max_y = len(map)
    print(' ' * 2, end='|')
    for i in range(max_x): print(f'{i:^3}', end='|')
    i = 0
    print()
    for row in range(max_y):
        print(f'{i:<2}', end='|')
        for col in range(max_x):
            if (col, row) in positions:
                print(f'{map[row][col]:*^4}', end = '')
            else:
                print(f'{map[row][col]:^4}', end = '')
        print()
        i += 1
    print()
    print(' '.join(refer))

ACTION_GO_LEFT = 0
ACTION_GO_RIGHT = 1
ACTION_GO_UP = 2
ACTION_GO_DOWN = 3
ACTION_FREE = 4
ACTION_CRAFT = 5
ilp = 0
HOST = "localhost"
PORT = 1111
if len(sys.argv) == 3:
    HOST = str(sys.argv[1])
    PORT = int(sys.argv[2])

# load json and create model
json_file = open('../../Training/TrainedModels/DQNmodel_20200806-0412_ep6900.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
DQNAgent = model_from_json(loaded_model_json)
# load weights into new model
DQNAgent.load_weights('../../Training/TrainedModels/DQNmodel_20200806-0412_ep6900.h5')
print("Loaded model from disk")
status_map = {0: "STATUS_PLAYING", 1: "STATUS_ELIMINATED_WENT_OUT_MAP", 2: "STATUS_ELIMINATED_OUT_OF_ENERGY",
                  3: "STATUS_ELIMINATED_INVALID_ACTION", 4: "STATUS_STOP_EMPTY_GOLD", 5: "STATUS_STOP_END_STEP"}
action_ls = []
try:
    # Initialize environment
    minerEnv = MinerEnv(HOST, PORT)
    minerEnv.start()  # Connect to the game
    minerEnv.reset()
    s = minerEnv.get_state()  ##Getting an initial state
    while not minerEnv.check_terminate():
        try:
            print(DQNAgent)
            action = np.argmax(DQNAgent.predict(s.reshape(1, len(s))))  # Getting an action from the trained model
            action_ls.append(action)
            print("next action = ", action)
            print('the array %s' %DQNAgent.predict(s.reshape(1, len(s))))
            minerEnv.step(str(action))  ############ Performing the action in order to obtain the new state
            s_next = minerEnv.get_state()  ############# Getting a new state
            s = s_next
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("Finished.")
            break
    print(status_map[minerEnv.state.status])
    print_map(minerEnv.init_map, action_ls)
except Exception as e:
    import traceback
    traceback.print_exc()
print("End game.")
