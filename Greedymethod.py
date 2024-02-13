import numpy as np
from matplotlib.path import Path
import matplotlib.pyplot as plt
import scipy
import pulp

temp=np.load('Lab2.npy')
freq=temp[:,:, 1]
pathnumber=temp[:,:,2]
skeleton=temp[:,:,0]

av=0

res=np.nonzero((skeleton==2))
res=list(zip(res[0],res[1]))
street=dict()

for i in res:
    if (pathnumber[i[0]][i[1]] in street.keys()):
        itemofpath = list()
        itemofpath.extend(street[pathnumber[i[0]][i[1]]])
        itemofpath.append([i[0], i[1]])
        street[pathnumber[i[0]][i[1]]] = itemofpath
    else:
        point = list()
        point.append([i[0], i[1]])
        street[pathnumber[i[0]][i[1]]] = point
Wcam=6
hcam1=10
hcam2=25
hcam3=40
lb=10
c1=1
c2=4
c3=8
cinstall=2
streetcam=dict()
x3=0
x2=0
x1=0
Arcpath=[11]
for key in street.keys():
    if (key == 0):
        continue
    lt=len(street[key])
    if(lt>hcam3):
        lt1 = lt % hcam3
        x3=lt//hcam3
    else:lt1=lt
    if (lt1 > hcam2):
        lt2 = lt1 % hcam2
        x2 = lt1 // hcam2
    else:lt2 = lt1
    if (lt2 > hcam1):
       x1 = lt2 // hcam1
    else:x1=0

    streetcam[key]=[x1,x2,x3]
    x3=0
    x2=0
    x1=0
img = np.zeros(temp.shape[0] * temp.shape[1] ).astype(int).reshape(temp.shape[0], temp.shape[1])

x, y = np.mgrid[:img.shape[0], :img.shape[1]]
allpoints = np.vstack((x.ravel(), y.ravel())).T

for key in street.keys():
    if (key == 0):
        continue

    r = street[key]
    points1 = list()
    points1.extend(r)
    l = len(points1)
    if (l <= hcam1):
        continue
    plt.text(points1[int(l/2)][1] , points1[int(l/2)][0], s=str(key))
    numcam = streetcam[key]
    numberofcam=list()

    for i2 in range(int(numcam[0])):
        numberofcam.append(hcam1)

    for i2 in range(int(numcam[1])):
        numberofcam.append(hcam2)

    for i2 in range(int(numcam[2])):
        numberofcam.append(hcam3)
    if key in Arcpath:
        numberofcam=list()
        for i in range(l//hcam1):
            numberofcam.append(hcam1)


    IDofCamInStreet = (200000 + key) * 1000
    IDofVisiblePoint = (100000 + key) * 1000
    cp=0
    c=0
    for i3 in numberofcam :
        Hcam=i3
        x1 = points1[cp][0]
        y1 = points1[cp][1]
        if (cp+Hcam)<l:
            x2 = points1[cp + Hcam][0]
            y2 = points1[cp + Hcam ][1]
        else:
            x2 = points1[l-1][0]
            y2 = points1[l-1][1]
        cp = cp + Hcam
        dx = x2 - x1
        dy = y2 - y1
        c=c+1
        if (dx != 0):
            slope = dy / dx
        else:
            slope = 100

        if (slope <= 5 and slope >= 0.2):

            vertices = np.asarray([(x1+Wcam, y1-Wcam),(x2+Wcam, y2-Wcam),
                               (x2-Wcam, y2+Wcam),(x1-Wcam, y1+Wcam)])
            path = Path(vertices)

            mask = path.contains_points(allpoints)

            path_points = allpoints[np.where(mask)]
            #
            for i in path_points:
                temp[i[0]][i[1]][3] = IDofVisiblePoint+c

            temp[x1][y1][3] = IDofCamInStreet+c
            plt.scatter(y1, x1, s=15, c='green', alpha=1, marker="X")



        elif (slope >= -5 and slope <= -0.2):

            vertices = np.asarray([(x1 - Wcam, y1 - Wcam), (x2 - Wcam, y2 - Wcam),
                               (x2 + Wcam, y2 + Wcam), (x1 + Wcam, y1 + Wcam)])
            path = Path(vertices)

            mask = path.contains_points(allpoints)

            path_points = allpoints[np.where(mask)]

            for i in path_points:
                temp[i[0]][i[1]][3] = IDofVisiblePoint+c

            temp[x1][y1][3] = IDofCamInStreet+c
            plt.scatter(y1, x1, s=15, c='green', alpha=1, marker="X")



        elif (slope < .2 and slope > -.2):

            vertices = np.asarray([(x1 , y1- Wcam), (x2 , y2- Wcam),
                               (x2 , y2+ Wcam ), (x1 , y1+ Wcam )])
            path = Path(vertices)

            mask = path.contains_points(allpoints)

            path_points = allpoints[np.where(mask)]

            for i in path_points:
                temp[i[0]][i[1]][3] = IDofVisiblePoint+c

            temp[x1][y1][3] = IDofCamInStreet+c
            plt.scatter(y1, x1, s=15, c='green', alpha=1, marker="X")


        elif (slope < -5 or slope > 5 or dx == 0):

            vertices = np.asarray([(x1-Wcam , y1), (x2 - Wcam, y2),
                               (x2+ Wcam , y2), (x1 + Wcam, y1)])
            path = Path(vertices)

            mask = path.contains_points(allpoints)
            path_points = allpoints[np.where(mask)]

            for i in path_points:
                temp[i[0]][i[1]][3] = IDofVisiblePoint+c

            temp[x1][y1][3] = IDofCamInStreet+c
            plt.scatter(y1, x1, s=15, c='green', alpha=1, marker="X")

v1=temp[:,:,3]
v2=np.where(v1>100000000)
vis=list(zip(v2[0],v2[1]))

v3=np.where(v1>200000000)
viscam=list(zip(v3[0],v3[1]))

x1total=0
x2total=0
x3total=0
for  key in streetcam.keys():
    x1total=x1total+streetcam[key][0]
    x2total = x1total + streetcam[key][1]
    x3total = x1total + streetcam[key][2]
print(x1total,x2total,x3total)
print((x1total*(c1+cinstall)),(x2total*(c2+cinstall)),(x3total*(c3+cinstall)))

maze1 = np.zeros(temp.shape[0] * temp.shape[1] * 4).reshape((temp.shape[0], temp.shape[1],4))
o1=list(zip(np.where(skeleton == 1)[0], np.where(skeleton == 1)[1]))
o0=list(zip(np.where(skeleton == 0)[0], np.where(skeleton == 0)[1]))
o2=list(zip(np.where(skeleton == 2)[0], np.where(skeleton == 2)[1]))
o3=list(zip(np.where(skeleton == 3)[0], np.where(skeleton == 3)[1]))
for i in o1:
    maze1[i[0]][i[1]][0]=.2;    maze1[i[0]][i[1]][1]=.2;    maze1[i[0]][i[1]][2]=.2;    maze1[i[0]][i[1]][3]=.5
for i in o0:
    maze1[i[0]][i[1]][0]=1;    maze1[i[0]][i[1]][1]=1;    maze1[i[0]][i[1]][2]=0;    maze1[i[0]][i[1]][3]=.2
for i in o3:
    maze1[i[0]][i[1]][0]=0;    maze1[i[0]][i[1]][1]=0;    maze1[i[0]][i[1]][2]=1;    maze1[i[0]][i[1]][3]=1
for i in o2:
    maze1[i[0]][i[1]][0]=1;    maze1[i[0]][i[1]][1]=0;    maze1[i[0]][i[1]][2]=0;    maze1[i[0]][i[1]][3]=1
for i in vis:
    maze1[i[0]][i[1]][0]=1;    maze1[i[0]][i[1]][1]=1;    maze1[i[0]][i[1]][2]=0;    maze1[i[0]][i[1]][3]=1
for i in viscam:
    maze1[i[0]][i[1]][0]=0;    maze1[i[0]][i[1]][1]=0;    maze1[i[0]][i[1]][2]=0;    maze1[i[0]][i[1]][3]=1

plt.imshow(maze1)
plt.show()
