import json
import numpy as np
temp = np.load('Lab1.npy')

maze = np.zeros(temp.shape[0]*temp.shape[1]*8).astype(int).reshape((temp.shape[0], temp.shape[1],8))

for i in range(1,temp.shape[0]-1):
    for j in range(1,temp.shape[1]-1):
        maze[i][j][0]=temp[i][j][0]
        maze[i][j][2] = temp[i][j][2]
print(maze)
with open('14path.json', "r") as r:
    response = r.read()
    response = response.replace('\n', '')
    response = response.replace('}{', '},{')
    response = "[" + response + "]"
print("convert to array done!")
parsed_json=json.loads(response)
print("load string done!")
list1 = []
for item in parsed_json:
    ii=item['path']

    for i1 in ii:
        maze[i1[0]][i1[1]][1]=maze[i1[0]][i1[1]][1]+1.0
for i in range(temp.shape[0]):
    for j in range(temp.shape[1]):
        if(maze[i][j][1] !=0):
            temp[i][j][1]=maze[i][j][1]
            print(i, j, maze[i][j][1])
np.save('Lab2.npy',temp)
print("save")

