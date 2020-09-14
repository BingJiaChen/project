import numpy as np
import math
import time
import matplotlib.pyplot as plt
import random

def multiAnchorPositioning(anchor_list, distance, last_pos, h):
    l_rate_x = 1
    l_rate_y = 1
    l = int(len(anchor_list)*(len(anchor_list)-1)/2)
    x_p1 = np.zeros((l, 1))
    y_p1 = np.zeros((l, 1))
    x_p2 = np.zeros((l, 1))
    y_p2 = np.zeros((l, 1))
    r_p = np.zeros((l, 1))
    counter = 0
    for i in range(len(anchor_list)-1):
        for j in np.arange(i+1, len(anchor_list)):
            x_p1[counter] = (anchor_list[i][0])
            y_p1[counter] = (anchor_list[i][1])
            x_p2[counter] = (anchor_list[j][0])
            y_p2[counter] = (anchor_list[j][1])
            r_p[counter] = (distance[i]-distance[j])
            counter = counter + 1
    x = last_pos[0]
    y = last_pos[1]
    Dx = 0
    Dy = 0
    vx = 0
    vy = 0
    eps = 1e-6
    beta = 0.9
    while True:
        x_old = x
        y_old = y
        f = np.sqrt((x - x_p1) ** 2 + (y - y_p1) ** 2 + h**2)-np.sqrt((x - x_p2) ** 2 + (y - y_p2) ** 2 + h**2)-r_p
        f_partial_x = 2*np.sum(f*((x - x_p1)/np.sqrt((x - x_p1)** 2 + (y - y_p1)** 2 + h**2) - (x - x_p2)/np.sqrt((x - x_p2)** 2 + (y - y_p2)** 2 + h**2)), axis = 0)[0]
        f_partial_y = 2*np.sum(f*((y - y_p1)/np.sqrt((x - x_p1)** 2 + (y - y_p1)** 2 + h**2) - (y - y_p2)/np.sqrt((x - x_p2)** 2 + (y - y_p2)** 2 + h**2)), axis = 0)[0]
        vx = beta*vx + (1-beta)*(f_partial_x**2)
        vy = beta*vy + (1-beta)*(f_partial_y**2)
        delta_x = -(np.sqrt(Dx+eps)/np.sqrt(vx+eps))*f_partial_x
        delta_y = -(np.sqrt(Dy+eps)/np.sqrt(vy+eps))*f_partial_y
        Dx = beta*Dx + (1-beta)*(delta_x**2)
        Dy = beta*Dy + (1-beta)*(delta_y**2)
        x = x + delta_x
        y = y + delta_y
        print(x,y)
        est = np.sqrt((x-x_old)**2+(y-y_old)**2)
        if est < 0.000001:
            break
    Ans = np.array([x,y])
    return Ans


#----- Example -----
if __name__ == '__main__':
    est=0
    spend_time=0
    interval = np.linspace(0,2*np.pi,100)
    estimated_tag_pos = np.array([1e-6,1e-6])
    times = 100
    for i in range(times):
        anchor_pos = np.array([[0, 0],  #Acnrho0(Center)
                            [8, 0],  #Anchor1
                            [8, 6],  #Anchor2
                            [0, 6]]) #Anchor3
        # tag_pos = np.array([4+5*np.cos(interval[i]),3+5*np.sin(interval[i])])
        tag_pos = np.array([4,3])
        # tag_pos = np.array([100+31.85*np.cos(interval[i]),100+31.85*np.sin(interval[i])])
        true_dist = np.linalg.norm(anchor_pos - tag_pos, axis = 1) #計算Tag到各Anchor的距離
        true_dist += np.random.normal(0, 0.06, 4) #加入誤差
        true_dist_diff = true_dist - true_dist[0] #計算距離差(相對於Center)
        # last_pos = estimated_tag_pos #Gradient Descent的起點，注意不能為(0, 0)
        last_pos = np.array([1e-6,1e-6])
        h = 0 #高度補償
        flag = time.time()
        estimated_tag_pos = multiAnchorPositioning(anchor_pos, true_dist, last_pos, h)
        t = time.time() - flag
        # plt.plot(estimated_tag_pos[0],estimated_tag_pos[1],"bo")
        est += np.sqrt((estimated_tag_pos[0]-tag_pos[0])**2+(estimated_tag_pos[1]-tag_pos[1])**2)
        spend_time+=t
        print("%d times:"%(i+1),t,"s")
    print("average error: ",est/times,"m")
    print("average time",spend_time/times,"s")