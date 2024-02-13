import numpy as np
from numpy.random import randint
from numpy.random import choice
from itertools import product as cartesian_product
from Astar_solver import AstarSolver
import matplotlib.pyplot as plt
import gym
import statistics
from maze import MazeEnv
import json




class MapMazeGenerator():
    def __init__(self):
        self.maze = self._generate_maze()
        self.mazeP= self._generate_mazeP()

    def get_maze(self):
        return self.maze

    def get_mazeP(self):
        return self.mazeP

    def _generate_maze(self):

        temp = np.load('Lab1.npy')
        maze = np.ones([temp.shape[0], temp.shape[1]])
        for i in range(1,temp.shape[0]-1):
            for j in range(1,temp.shape[1]-1):
                if( temp[i][j][0]==2 or temp[i][j][0]==3):maze[i][j]=0
        return maze

    def _generate_mazeP(self):

        temp3 = np.load('Labw.npy')
        mazeP = np.ones([temp3.shape[0], temp3.shape[1]])
        counter_10 = counter_60 = counter_50 =counter_80= 0
        for i in range(1, temp3.shape[0] - 1):
            for j in range(1, temp3.shape[1] - 1):
                if (temp3[i][j][4] == .9):
                    counter_10 += 1
                    mazeP[i][j] = 10
                elif (temp3[i][j][4] == .5):
                    mazeP[i][j] = 60
                    counter_60 += 1
                elif (temp3[i][j][4] == .2):
                    mazeP[i][j] = 80
                    counter_80 += 1

                elif (temp3[i][j][4] == 1):
                    mazeP[i][j] = 50
                    counter_50 += 1

        return mazeP

    def sample_state(self):


        free_space = np.where(self.maze == 0)

        free_space = list(zip(free_space[0], free_space[1]))

        # Sample indices for initial state and goal state
        init_idx, goal_idx = np.random.choice(len(free_space), size=2, replace=False)

        # Convert initial state to a list, goal states to list of list
        init_state = list(free_space[init_idx])
        goal_states = [list(free_space[goal_idx])]  # TODO: multiple goals

        return init_state, goal_states


    def GetSample_State(self):
        free_space = np.where(self.maze == 0)
        free_space = list(zip(free_space[0], free_space[1]))

        probability_distribution=list()
        for i in range(len(free_space)):
            probability_distribution.append(self.mazeP[free_space[i][0]][free_space[i][1]])
        number_of_items_to_pick = 2
        p1 = sum(probability_distribution)
        probability_distribution = [number / p1 for number in probability_distribution]
        init_idx, goal_idx  = choice(len(free_space), number_of_items_to_pick, p=probability_distribution)
        init_state = list(free_space[init_idx])
        goal_states = [list(free_space[goal_idx])]
        return init_state,  goal_states

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self)

def solvemaze(maze, action_type='Moore', render_trace=False, gif_file='video.gif'):
    env = MazeEnv(maze, action_type=action_type, render_trace=render_trace)
    env.reset()
    # Solve maze by A* search from current state to goal
    solver = AstarSolver(env, env.goal_states[0])
    if not solver.solvable():
        #raise Error('The maze is not solvable given the current state and the goal state')
        return "NOT"
    else:

        return list(solver.get_states())

my_dictionary = dict()

maze=MapMazeGenerator()

pointlist=list()
my_dictionary=dict()
jsonFile = open("path.json", "w")


for i2 in range(40000):
    temp=solvemaze(maze, action_type='Moore', render_trace=False, gif_file='test1_block_maze.gif')
    if (temp== "NOT"):
        print(" Not solve")
    else:
        l1 = len(temp)
        #origin=temp[i], destination=temp[l1 - 1],
        my_dictionary = dict(origin=temp[0], destination=temp[l1 - 1], path=temp, number=str(l1))
        jsonString = json.dumps(my_dictionary, cls=NpEncoder)
        jsonFile.write(jsonString)
        pointlist.extend(temp)

jsonFile.close()
