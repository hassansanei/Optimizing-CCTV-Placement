import numpy as np
from numpy import ma
import os
import matplotlib

import cv2
import matplotlib.pyplot as plt
#import data_utils
from skimage.morphology import skeletonize
#from data_utils import affinity_utils
import sknw
gaussian_road_mask = cv2.imread("Lab.png",0)
##############################

gaussian_road_mask = gaussian_road_mask.astype(np.float)
threshold = 0.76
road_mask = np.copy(gaussian_road_mask/255.0)
road_mask[road_mask > threshold] = 1
road_mask[road_mask <= threshold] = 0
maze = np.zeros(gaussian_road_mask.shape[0] * gaussian_road_mask.shape[1] * 8).astype(int).reshape((gaussian_road_mask.shape[0], gaussian_road_mask.shape[1], 8))
# # skeletonize the binary road mask
skeleton = skeletonize(road_mask).astype(np.uint16)
# build graph from skeleton
graph = sknw.build_sknw(skeleton)

# draw image
plt.figure(figsize=(6, 6))
#plt.imshow(gaussian_road_mask)
#plt.imshow(road_mask)
#plt.imshow(skeleton)

# draw edges by pts
cont=1
for (s, e) in graph.edges():
    ps = graph[s][e]['pts']
    for i in range(len(ps)):
        maze[ps[i][0]][ps[i][1]][2] = cont
    cont=cont+1
    #print(ps)

#    plt.plot(ps[:, 1], ps[:, 0], 'yellow')

# draw node by o
nodes = graph.nodes()
ps = np.array([nodes[i]['o'] for i in nodes])

#plt.plot(ps[:, 1], ps[:, 0], 'r.')

for i in range(1,gaussian_road_mask.shape[0]-1):
    for j in range(1,gaussian_road_mask.shape[1]-1):
        #if(skeleton[i][j]!=0):print(skeleton[i][j])
        if(gaussian_road_mask[i][j]==0 ):maze[i][j][0]=1
        if (gaussian_road_mask[i][j] == 1): maze[i][j][0] = 0
        if (skeleton[i][j] == 1):maze[i][j][0] = 2
for i in range(len(ps)):
    maze[ps[i][0]][ps[i][1]][0]=3
color1=int('ff',base=16)/255
color2=int('cb',base=16)/255
color3=int('9a',base=16)/255
color4=int('cd',base=16)/255
color5=int('32',base=16)/255

maze1 = np.zeros(maze.shape[0] * maze.shape[1] * 4).reshape((maze.shape[0], maze.shape[1], 4))
for i in range(1, maze.shape[0] - 1):
    for j in range(1, maze.shape[1] - 1):
        if (maze[i][j][0] == 1):
            maze1[i][j] [0]= color1
            maze1[i][j][1] = color1
            maze1[i][j][2] = color2
            maze1[i][j][3] = 1
        if (maze[i][j][0] == 0):
            maze1[i][j] [0]= color3
            maze1[i][j][1] = color4
            maze1[i][j][2] = color5
            maze1[i][j][3] = 1
        if (maze[i][j][0] == 2):
            maze1[i][j] [0]= 0
            maze1[i][j][1] = 0
            maze1[i][j][2] = 0
            maze1[i][j][3] = 1
        if (maze[i][j][0] == 3):
            maze1[i][j] [0]= int(255/255)
            maze1[i][j][1] = 0
            maze1[i][j][2] = 0
            maze1[i][j][3] = 1
        if (maze[i][j][2] == 20 or maze[i][j][2] == 19):
            maze1[i][j][0] = int(255 / 255)
            maze1[i][j][1] = 0
            maze1[i][j][2] = int(255 / 255)
            maze1[i][j][3] = 1

#np.save('kashan3.npy', maze)
plt.imshow(maze1)
plt.show()

