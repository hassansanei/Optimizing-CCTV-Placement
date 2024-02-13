import logging
from enum import Enum, IntEnum
import random
import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime
import matplotlib.animation as animation


class Cell(IntEnum):
    EMPTY = 0  # indicates empty cell where the agent can move to
    OCCUPIED = 1  # indicates cell which contains a wall and cannot be entered
    CURRENT = 2  # indicates current cell of the agent
    CAMERA = 3

class Action(IntEnum):
    MOVE_LEFT = 0
    MOVE_RIGHT = 1
    MOVE_UP = 2
    MOVE_DOWN = 3
    MOVE_UPRIGHT = 4
    MOVE_UPLEFT = 5
    MOVE_DOWNRIGHT = 6
    MOVE_DOWNLEFT = 7




class Status(Enum):
    WIN = 0
    LOSE = 1
    PLAYING = 2


class RMaze1:
    actions = [Action.MOVE_LEFT, Action.MOVE_RIGHT, Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_UPRIGHT
        , Action.MOVE_UPLEFT, Action.MOVE_DOWNRIGHT, Action.MOVE_DOWNLEFT]  # all possible actions

    reward_exit = 10.0  # reward for reaching the exit cell
    penalty_move = -0.05  # penalty for a move which did not result in finding the exit cell
    penalty_visited = -0.25  # penalty for returning to a cell which was visited earlier
    penalty_impossible_move = -0.75  # penalty for trying to enter an occupied cell or moving out of the maze
    penalty_camera = -0.5


    def __init__(self, maze, start_cell, exit_cell):

        self.maze = maze

        self.__current_cell = start_cell


        self.__minimum_reward = -0.008 * self.maze.size  # stop game if accumulated reward is below this threshold

        nrows, ncols = self.maze.shape
        self.cells = [(col, row) for col in range(ncols) for row in range(nrows)]
        self.camera = [(col, row) for col in range(ncols) for row in range(nrows) if self.maze[row, col] == Cell.CAMERA]

        self.empty = [(col, row) for col in range(ncols) for row in range(nrows) if self.maze[row, col] == Cell.EMPTY]
        self.empty.extend(self.camera)

        self.start_cell = start_cell
        self.__exit_cell = exit_cell
        if (self.__exit_cell not in self.empty):
            print(" self.__exit_cell not in self.empty")

        self.empty.remove(self.__exit_cell)

        # Check for impossible maze layout
        if self.__exit_cell not in self.cells:
            raise Exception("Error: exit cell at {} is not inside maze".format(self.__exit_cell))
        if self.maze[self.__exit_cell[::-1]] == Cell.OCCUPIED:
            raise Exception("Error: exit cell at {} is not free".format(self.__exit_cell))
        self.Q=dict()
        self.reset(start_cell)
    def getQ(self):
        actions1 = [Action.MOVE_LEFT, Action.MOVE_RIGHT, Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_UPRIGHT
            , Action.MOVE_UPLEFT, Action.MOVE_DOWNRIGHT, Action.MOVE_DOWNLEFT]  # all possible actions

        for i in self.empty:

            res=self.__possible_actions(i)
            for j in actions1:
                if(j in res):
                    self.Q[i,j]=0
                else:
                    self.Q[i,j]=-200
        return self.Q

    def reset(self, start_cell):

        if start_cell not in self.cells:
            raise Exception("Error: start cell at {} is not inside maze".format(start_cell))
        if self.maze[start_cell[::-1]] == Cell.OCCUPIED:
            raise Exception("Error: start cell at {} is not free".format(start_cell))
        if start_cell == self.__exit_cell:
            raise Exception("Error: start- and exit cell cannot be the same {}".format(start_cell))

        self.__previous_cell = self.__current_cell = start_cell
        self.__total_reward = 0.0  # accumulated reward
        self.__visited = set()  # a set() only stores unique values

        return self.__observe()

    def step(self, action):

        reward = self.__execute(action)
        self.__total_reward += reward
        status = self.__status()
        state = self.__observe()
        logging.debug("action: {:10s} | reward: {: .2f} | status: {}".format(Action(action).name, reward, status))

        return state, reward, status

    def __execute(self, action):

        possible_actions = self.__possible_actions(self.__current_cell)

        if not possible_actions:
            reward = self.__minimum_reward - 1  # cannot move anywhere, force end of game
        elif action in possible_actions:
            col, row = self.__current_cell
            if action == Action.MOVE_LEFT:
                col -= 1
            elif action == Action.MOVE_UP:
                row -= 1
            if action == Action.MOVE_RIGHT:
                col += 1
            elif action == Action.MOVE_DOWN:
                row += 1
            if action == Action.MOVE_UPRIGHT:
                row -= 1
                col += 1
            elif action == Action.MOVE_UPLEFT:
                row -= 1
                col -= 1
            if action == Action.MOVE_DOWNRIGHT:
                row += 1
                col += 1
            elif action == Action.MOVE_DOWNLEFT:
                row += 1
                col -= 1

            self.__previous_cell = self.__current_cell
            self.__current_cell = (col, row)


            if self.__current_cell == self.__exit_cell:
                reward = RMaze1.reward_exit  # maximum reward when reaching the exit cell
            elif self.__current_cell in self.__visited:
                reward = RMaze1.penalty_visited
            elif self.__current_cell in self.camera:
                reward = RMaze1.penalty_camera

            else:
                reward = RMaze1.penalty_move  # penalty for a move which did not result in finding the exit cell

            self.__visited.add(self.__current_cell)
        else:
            reward = RMaze1.penalty_impossible_move  # penalty for trying to enter an occupied cell or move out of the maze

        return reward

    def __possible_actions(self, cell=None):

        if cell is None:
            col, row = self.__current_cell
        else:
            col, row = cell

        possible_actions = RMaze1.actions.copy()  # initially allow all

        # now restrict the initial list by removing impossible actions
        nrows, ncols = self.maze.shape
        if row == 0 or (row > 0 and self.maze[row - 1, col] == Cell.OCCUPIED):
            possible_actions.remove(Action.MOVE_UP)
        if row == 0 or col == 0 or (row > 0 and self.maze[row - 1, col - 1] == Cell.OCCUPIED):
            possible_actions.remove(Action.MOVE_UPLEFT)
        if row == 0 or col == ncols - 1 or (row > 0 and self.maze[row - 1, col + 1] == Cell.OCCUPIED):
            possible_actions.remove(Action.MOVE_UPRIGHT)
        if row == nrows - 1 or (row < nrows - 1 and self.maze[row + 1, col] == Cell.OCCUPIED):
            possible_actions.remove(Action.MOVE_DOWN)
        if row == nrows - 1 or col == 0 or (row < nrows - 1 and self.maze[row + 1, col - 1] == Cell.OCCUPIED):
            possible_actions.remove(Action.MOVE_DOWNLEFT)
        if row == nrows - 1 or col == ncols - 1 or (row < nrows - 1 and self.maze[row + 1, col + 1] == Cell.OCCUPIED):
            possible_actions.remove(Action.MOVE_DOWNRIGHT)
        if col == 0 or (col > 0 and self.maze[row, col - 1] == Cell.OCCUPIED):
            possible_actions.remove(Action.MOVE_LEFT)
        if col == ncols - 1 or (col < ncols - 1 and self.maze[row, col + 1] == Cell.OCCUPIED):
            possible_actions.remove(Action.MOVE_RIGHT)

        return possible_actions

    def __status(self):

        if self.__current_cell == self.__exit_cell:
            return Status.WIN

        if self.__total_reward < self.__minimum_reward:  # force end of game after too much loss
            return Status.LOSE

        return Status.PLAYING

    def __observe(self):

        return np.array([[*self.__current_cell]])


class QTableModel():

    def __init__(self, game,**kwargs):


        self.environment = game
        self.Q = self.environment.getQ()
        #self.Q = dict()


    def train(self, stop_at_convergence=False, **kwargs):

        discount = kwargs.get("discount", 0.90)
        exploration_rate = kwargs.get("exploration_rate", 0.10)
        exploration_decay = kwargs.get("exploration_decay", 0.995)
        learning_rate = kwargs.get("learning_rate", 0.10)

        episodes=500

        cumulative_reward = 0
        win_history = []

        maxcumulative_reward = -1000000000
        cumulative_reward_history = []
        # training starts here
        for episode in range(1, episodes + 1):
            start_cell=self.environment.start_cell
            state = self.environment.reset(start_cell)
            state = tuple(state.flatten())
            cumulative_reward = 0
            visited = set()

            while True:
                if np.random.random() < exploration_rate:
                    action = random.choice(self.environment.actions)
                else:
                    action = self.predict(state)


                next_state, reward, status = self.environment.step(action)
                next_state = tuple(next_state.flatten())
                visited.add(next_state)
                cumulative_reward += reward


                max_next_Q = max([self.Q.get((next_state, a), 0.0) for a in self.environment.actions])

                self.Q[(state, action)] += learning_rate * (reward + discount * max_next_Q - self.Q[(state, action)])

                if status in (Status.WIN, Status.LOSE):
                    if (status ==Status.WIN):
                        if (cumulative_reward > maxcumulative_reward):
                            savevisited = list()
                            savevisited.extend(visited)
                            maxcumulative_reward = cumulative_reward
                            cumulative_reward_history.append(cumulative_reward)

                    break

                state = next_state


            exploration_rate *= exploration_decay  # explore less as training progresses
        return cumulative_reward_history, win_history,savevisited, episode

    def q(self, state):
        """ Get q values for all actions for a certain state. """
        if type(state) == np.ndarray:
            state = tuple(state.flatten())

        return np.array([self.Q.get((state, action), 0.0) for action in self.environment.actions])

    def predict(self, state):

        q = self.q(state)

        logging.debug("q[] = {}".format(q))

        actions = np.nonzero(q == np.max(q))[0]  # get index of the action(s) with the max value
        return random.choice(actions)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self)



temp = np.load('Lab2.npy')
maze = np.ones([temp.shape[0], temp.shape[1]])
for i in range(1,temp.shape[0]-1):
    for j in range(1,temp.shape[1]-1):
        cellstate=temp[i][j][0]
        cellcamvisible=temp[i][j][3]
        if(cellstate==2 or cellstate==3):maze[i][j] = 0
        if ((cellstate == 2 or cellstate == 3)and
                (cellcamvisible==10 or cellcamvisible==20)): maze[i][j] = 3
dic1={(187, 25): [[1501, 349], [351, 697], 1185], (43, 197): [[422, 647], [1506, 717], 1151], (43, 188): [[422, 647], [1527, 349], 1128], (29, 160): [[453, 1111], [1221, 492], 1136], (29, 182): [[436, 1072], [1344, 444], 1263], (197, 51): [[1541, 703], [429, 651], 1180], (31, 187): [[384, 686], [1515, 342], 1193], (29, 183): [[452, 1110], [1339, 425], 1272], (197, 31): [[1531, 707], [376, 676], 1188], (0, 2): [[1305, 452], [36, 656], 1297], (25, 157): [[351, 698], [1170, 211], 1262], (182, 18): [[1321, 451], [351, 876], 1105], (187, 51): [[1523, 339], [442, 648], 1107], (157, 24): [[1119, 197], [353, 638], 1271], (178, 29): [[1314, 443], [442, 1086], 1247], (8, 183): [[246, 389], [1332, 424], 1111], (30, 176): [[378, 666], [1381, 926], 1150], (157, 51): [[1135, 202], [438, 649], 1184], (30, 187): [[383, 665], [1518, 341], 1165], (156, 30): [[1084, 239], [404, 659], 1112], (117, 133): [[766, 812], [908, 91], 1142], (2, 177): [[24, 656], [1282, 460], 1290], (197, 43): [[1521, 710], [420, 641], 1177], (51, 187): [[445, 647], [1534, 333], 1119], (25, 187): [[352, 692], [1530, 335], 1213], (30, 197): [[388, 663], [1523, 710], 1190], (128, 117): [[830, 89], [805, 804], 1094], (25, 188): [[349, 704], [1533, 347], 1218], (71, 198): [[560, 623], [1443, 1073], 1167], (24, 176): [[362, 645], [1359, 927], 1144], (187, 43): [[1529, 336], [421, 644], 1139], (186, 29): [[1342, 429], [420, 1035], 1240], (188, 0): [[1492, 366], [370, 671], 1139], (157, 19): [[1310, 394], [310, 33], 1251], (117, 128): [[819, 802], [828, 97], 1101], (182, 9): [[1342, 447], [184, 587], 1239], (188, 30): [[1524, 351], [371, 668], 1168], (187, 0): [[1501, 349], [370, 669], 1157], (129, 25): [[984, 160], [351, 700], 1132], (197, 30): [[1537, 705], [383, 665], 1222], (9, 177): [[182, 578], [1293, 457], 1200], (157, 25): [[1157, 207], [351, 693], 1267], (197, 25): [[1541, 703], [352, 701], 1235], (188, 31): [[1533, 347], [388, 692], 1202], (113, 1): [[1008, 568], [26, 518], 1100], (14, 160): [[277, 478], [1253, 477], 1100], (140, 0): [[913, 784], [834, 65], 1152], (198, 51): [[1462, 1144], [464, 641], 1336], (25, 197): [[350, 703], [1515, 713], 1195], (58, 177): [[522, 1125], [1290, 459], 1125], (25, 156): [[364, 676], [1049, 197], 1181], (31, 197): [[373, 674], [1517, 712], 1161], (117, 0): [[810, 803], [837, 65], 1131], (25, 133): [[350, 703], [918, 93], 1100], (30, 156): [[379, 666], [1077, 231], 1199], (180, 18): [[1316, 436], [333, 837], 1147], (30, 188): [[385, 664], [1514, 355], 1149], (140, 3): [[982, 771], [75, 451], 1126], (170, 19): [[1097, 756], [326, 65], 1101], (183, 19): [[1333, 424], [305, 23], 1244], (51, 197): [[427, 652], [1537, 705], 1174], (176, 25): [[1338, 927], [353, 690], 1131], (3, 155): [[75, 457], [1045, 762], 1167], (183, 18): [[1322, 423], [331, 833], 1164], (160, 2): [[1212, 496], [7, 657], 1231], (3, 160): [[77, 444], [1211, 496], 1329], (188, 25): [[1523, 351], [347, 704], 1205], (2, 178): [[103, 653], [1314, 443], 1245], (2, 160): [[57, 655], [1251, 478], 1224], (187, 30): [[1531, 335], [389, 663], 1168], (176, 24): [[1376, 926], [348, 637], 1190], (43, 198): [[422, 650], [1429, 1018], 1249], (167, 7): [[1085, 774], [164, 491], 1102], (157, 43): [[1130, 200], [420, 642], 1204], (179, 18): [[1310, 453], [334, 840], 1130], (153, 19): [[1034, 760], [303, 20], 1157], (157, 30): [[1128, 200], [396, 661], 1220], (9, 156): [[171, 528], [1275, 453], 1245], (179, 2): [[1309, 453], [7, 657], 1330], (156, 24): [[1073, 226], [367, 662], 1166], (179, 8): [[1317, 452], [219, 398], 1107], (19, 182): [[308, 30], [1344, 443], 1234], (111, 133): [[734, 812], [900, 89], 1102], (30, 157): [[376, 666], [1162, 209], 1235], (180, 58): [[1319, 429], [527, 1124], 1162], (182, 29): [[1335, 452], [386, 957], 1137], (140, 128): [[920, 783], [829, 94], 1127], (186, 2): [[1343, 433], [29, 656], 1355], (2, 186): [[110, 653], [1344, 436], 1276], (140, 132): [[897, 787], [836, 79], 1121], (31, 198): [[377, 678], [1471, 1179], 1407], (51, 188): [[431, 651], [1516, 354], 1105], (0, 117): [[829, 74], [798, 805], 1102], (156, 51): [[1048, 196], [435, 650], 1108], (182, 19): [[1340, 449], [323, 58], 1201], (181, 29): [[1317, 443], [432, 1061], 1225], (1, 140): [[32, 519], [1026, 764], 1131], (178, 3): [[1315, 443], [76, 491], 1391], (3, 157): [[75, 449], [1319, 414], 1461], (19, 179): [[319, 52], [1313, 453], 1178], (51, 156): [[444, 648], [1061, 212], 1115], (51, 157): [[428, 652], [1183, 214], 1162], (18, 0): [[329, 830], [1342, 424], 1193], (182, 7): [[1326, 452], [165, 499], 1256], (26, 186): [[418, 481], [1342, 429], 1183], (0, 156): [[370, 671], [1058, 208], 1187], (179, 19): [[1309, 453], [305, 23], 1204], (2, 182): [[52, 655], [1327, 452], 1307], (177, 2): [[1284, 460], [16, 656], 1296], (155, 3): [[1069, 760], [76, 510], 1153], (176, 30): [[1345, 927], [388, 663], 1134], (18, 183): [[328, 828], [1324, 423], 1173], (29, 177): [[452, 1108], [1302, 453], 1222], (43, 187): [[422, 648], [1529, 336], 1139], (25, 198): [[365, 675], [1466, 1161], 1402], (8, 181): [[165, 417], [1318, 444], 1165], (188, 22): [[1349, 438], [329, 415], 1103], (177, 58): [[1297, 455], [529, 1123], 1126], (133, 98): [[910, 91], [721, 810], 1098], (4, 160): [[115, 522], [1241, 482], 1245], (156, 31): [[1068, 220], [377, 677], 1174], (167, 19): [[1086, 782], [332, 77], 1110], (19, 186): [[351, 115], [1342, 428], 1160], (181, 58): [[1319, 447], [543, 1119], 1138], (19, 140): [[308, 29], [1022, 765], 1137], (160, 29): [[1270, 469], [435, 1068], 1181], (179, 198): [[1318, 452], [1480, 1213], 1156], (0, 157): [[143, 431], [1321, 420], 1214], (179, 29): [[1311, 453], [452, 1109], 1232], (183, 16): [[1328, 423], [308, 577], 1109], (160, 58): [[1274, 467], [515, 1127], 1115], (175, 1): [[1137, 759], [2, 519], 1290], (19, 155): [[317, 47], [1038, 762], 1135], (152, 3): [[1034, 577], [76, 488], 1109], (178, 2): [[1315, 443], [42, 656], 1303], (186, 19): [[1344, 436], [315, 44], 1228], (177, 19): [[1294, 457], [310, 33], 1179], (177, 18): [[1278, 460], [335, 842], 1097], (2, 183): [[67, 655], [1338, 424], 1318], (58, 186): [[609, 1100], [1342, 428], 1106], (177, 29): [[1283, 460], [451, 1107], 1206], (14, 156): [[276, 474], [1263, 440], 1145], (160, 11): [[1274, 467], [221, 701], 1112], (25, 176): [[351, 696], [1383, 926], 1168], (9, 157): [[172, 531], [1321, 419], 1295], (183, 2): [[1339, 425], [80, 654], 1302], (180, 2): [[1316, 438], [20, 656], 1330], (51, 198): [[451, 645], [1479, 1210], 1412], (117, 132): [[817, 802], [841, 76], 1126], (198, 30): [[1464, 1152], [389, 663], 1416], (18, 182): [[341, 854], [1340, 449], 1147], (19, 183): [[325, 62], [1328, 423], 1198], (198, 102): [[1461, 1140], [705, 566], 1141], (18, 186): [[337, 847], [1344, 436], 1169], (76, 198): [[604, 601], [1452, 1107], 1208], (180, 29): [[1318, 432], [415, 1024], 1196], (183, 1): [[1325, 423], [47, 519], 1415], (2, 179): [[150, 651], [1306, 452], 1188], (58, 182): [[577, 1109], [1335, 452], 1115], (29, 181): [[422, 1039], [1317, 443], 1205], (19, 160): [[312, 37], [1217, 493], 1095], (19, 168): [[316, 46], [1089, 766], 1124], (18, 177): [[337, 846], [1285, 459], 1101], (4, 178): [[103, 521], [1315, 443], 1335], (133, 108): [[905, 90], [726, 816], 1101], (157, 9): [[1321, 419], [175, 549], 1279], (19, 180): [[337, 87], [1320, 428], 1162], (58, 160): [[507, 1130], [1273, 467], 1121], (19, 163): [[310, 33], [1073, 758], 1128], (101, 198): [[701, 580], [1474, 1189], 1196], (113, 3): [[1014, 570], [75, 457], 1118], (188, 51): [[1525, 350], [441, 648], 1099], (29, 180): [[431, 1060], [1321, 424], 1243], (19, 166): [[322, 57], [1089, 753], 1100], (26, 183): [[397, 481], [1336, 424], 1154], (164, 3): [[1073, 761], [75, 461], 1207], (2, 180): [[51, 655], [1316, 437], 1304], (182, 58): [[1341, 448], [554, 1116], 1145], (160, 16): [[1272, 468], [291, 521], 1094]}

jsonFile = open("pathRL.json", "w")
for key in dic1.keys():
    sample1=dic1[key]
    game = RMaze1(maze,(sample1[0][1],sample1[0][0]),(sample1[1][1],sample1[1][0]))
    visit=set()
    model =QTableModel(game)
    h, w,visit, _ = model.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=200,
                               stop_at_convergence=True)
    vis=list(set(visit))
    my_dictionary = dict()
    my_dictionary = dict(origin=sample1[0], destination=sample1[1], path=vis)
    jsonString = json.dumps(my_dictionary, cls=NpEncoder)
    jsonFile.write(jsonString)
    print("visit",vis)
