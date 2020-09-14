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
    wX = 1e-10
    wY = 1e-10
    beta = 0
    lamda = 0
    init = True
    d_x = 0
    d_y = 0
    H = np.ones((len(anchor_list)-1,1))
    G = np.ones((len(anchor_list)-1,2))
    G1 = np.ones((2,len(anchor_list)-1))
    W = np.ones((2,2))
    tag = np.array([x,y])
    Q = np.identity(len(anchor_list)-1)
    Q_inv = np.linalg.inv(Q)
    R = np.ones((len(anchor_list)-1,1))
    temp = np.ones(R.shape)
    flag = 0
    f = np.sqrt((x - x_p1) ** 2 + (y - y_p1) ** 2 + h**2)-np.sqrt((x - x_p2) ** 2 + (y - y_p2) ** 2 + h**2)-r_p
    f_sum = np.sum(f**2)
    f_partial_x = 2*np.sum(f*((x - x_p1)/np.sqrt((x - x_p1)** 2 + (y - y_p1)** 2 + h**2) - (x - x_p2)/np.sqrt((x - x_p2)** 2 + (y - y_p2)** 2 + h**2)), axis = 0)[0]
    f_partial_y = 2*np.sum(f*((y - y_p1)/np.sqrt((x - x_p1)** 2 + (y - y_p1)** 2 + h**2) - (y - y_p2)/np.sqrt((x - x_p2)** 2 + (y - y_p2)** 2 + h**2)), axis = 0)[0]
    for times in range(5):
        x_old = x
        y_old = y
        lamda = (f_sum)/(f_partial_x**2+f_partial_y**2)
        # print(lamda)
        d_x = -f_partial_x + beta*d_x
        d_y = -f_partial_y + beta*d_y
        x = x + lamda*(d_x)
        y = y + lamda*(d_y)
        GD_norm = np.sqrt(f_partial_x**2 + f_partial_y**2) 
        f = np.sqrt((x - x_p1) ** 2 + (y - y_p1) ** 2 + h**2)-np.sqrt((x - x_p2) ** 2 + (y - y_p2) ** 2 + h**2)-r_p
        f_sum = np.sum(f**2)
        f_partial_x = 2*np.sum(f*((x - x_p1)/np.sqrt((x - x_p1)** 2 + (y - y_p1)** 2 + h**2) - (x - x_p2)/np.sqrt((x - x_p2)** 2 + (y - y_p2)** 2 + h**2)), axis = 0)[0]
        f_partial_y = 2*np.sum(f*((y - y_p1)/np.sqrt((x - x_p1)** 2 + (y - y_p1)** 2 + h**2) - (y - y_p2)/np.sqrt((x - x_p2)** 2 + (y - y_p2)** 2 + h**2)), axis = 0)[0]
        beta = np.sqrt(f_partial_x**2 + f_partial_y**2)/GD_norm	
        tag = np.array([x,y])
        est = np.sqrt((x-x_old)**2+(y-y_old)**2)
        # print(x,y)
        
        if est < 0.1 or np.sqrt(f_partial_x**2+f_partial_y**2)<1e-7:
            break
    tag = np.array([x,y])
    # print(tag)
    for times in range(100):
        for i in range(len(anchor_list)-1):
            H[i] = np.array([distance[i+1]-distance[0]-(np.linalg.norm(anchor_list[i+1]-tag)-np.linalg.norm(anchor_list[0]-tag))])
            x1 = (anchor_list[0,0]-tag[0])/(np.linalg.norm(anchor_list[0]-tag))-(anchor_list[i+1,0]-tag[0])/(np.linalg.norm(anchor_list[i+1]-tag))
            x2 = (anchor_list[0,1]-tag[1])/(np.linalg.norm(anchor_list[0]-tag))-(anchor_list[i+1,1]-tag[1])/(np.linalg.norm(anchor_list[i+1]-tag))
            G[i] = np.array([x1,x2])
            if x1==0 or x2==0:
                return tag
            temp[i] = np.array([np.linalg.norm(anchor_list[i+1]-tag)-np.linalg.norm(anchor_list[0]-tag)])
        if flag<10:
            R = np.copy(temp)
        else:
            R = np.hstack((R,temp))  
            for j in range(len(anchor_list)-1):
                Q[j,j] = np.std(R[j])
            # Q = np.cov(R)
            Q_inv = np.linalg.pinv(Q)
        delta = np.dot(np.dot(np.dot(np.linalg.pinv(np.dot(G.T,np.dot(Q_inv,G))),G.T),Q_inv),H)
        # delta = np.dot(np.dot(np.linalg.inv(np.dot(G.T,G)),G.T),H)
        for j in range(len(anchor_list)-1):
            partial_xx = 1/np.linalg.norm(anchor_list[j+1]-tag) * (1-((anchor_list[j+1,0]-tag[0])/(np.linalg.norm(anchor_list[j+1]-tag)))**2) - 1/np.linalg.norm(anchor_list[0]-tag) * (1-((anchor_list[0,0]-tag[0])/(np.linalg.norm(anchor_list[0]-tag)))**2)
            partial_xy = -1/np.linalg.norm(anchor_list[j+1]-tag) * (anchor_list[j+1,0]-tag[0])/np.linalg.norm(anchor_list[j+1]-tag) * (anchor_list[j+1,1]-tag[1])/np.linalg.norm(anchor_list[j+1]-tag) + 1/np.linalg.norm(anchor_list[0]-tag) * (anchor_list[0,0]-tag[0])/np.linalg.norm(anchor_list[0]-tag) * (anchor_list[0,1]-tag[1])/np.linalg.norm(anchor_list[0]-tag)
            partial_yy = -1/np.linalg.norm(anchor_list[0]-tag) * (1-((anchor_list[0,1]-tag[1])/(np.linalg.norm(anchor_list[0]-tag)))**2) + 1/np.linalg.norm(anchor_list[j+1]-tag) * (1-((anchor_list[j+1,1]-tag[1])/(np.linalg.norm(anchor_list[j+1]-tag)))**2)
            W = np.array([[partial_xx,partial_xy],[partial_xy,partial_yy]])
            G1[:,j] = np.dot(W.T,delta)[:,0]
        G = G + 0.5*G1.T
        delta = np.dot(np.dot(np.dot(np.linalg.pinv(np.dot(G.T,np.dot(Q_inv,G))),G.T),Q_inv),H)
        if np.sqrt(delta.T[0,0]**2+delta.T[0,1]**2)>1e+2:
            pass
        else:
            tag = tag + delta.T[0]
        # print(flag)
        # print(tag)
        flag+=1
        if np.linalg.norm(delta.T[0])<0.001:
            break
    return tag


#----- Example -----
if __name__ == '__main__':
    est=0
    spend_time=0
    interval = np.linspace(0,2*np.pi,100)
    estimated_tag_pos = np.array([1e-6,1e-6])
    times = 100
    itr = 100
    alpha = 1.9
    for row in range(39):
        for col in range(31):
            est = 0
            spend_time = 0
            print("Test tag position: ",([-5+0.5*row,-5+0.5*col]))
            for i in range(times):
                z = []
                anchor_pos = np.array([[0, 0],  #Acnrho0(Center)
                                            [9, 0],  #Anchor1
                                            [9, 5],  #Anchor2
                                            [0, 5]]) #Anchor3
                # anchor_pos = np.array([[0, 0],  #Acnrho0(Center)
                #                         [200, 0],  #Anchor1
                #                         [200, 200],  #Anchor2 
                #                         [0, 200]]) #Anchor3
                # tag_pos = np.array([4+5*np.cos(interval[i]),3+5*np.sin(interval[i])])
                tag_pos = np.array( [-5+0.5*row,-5+0.5*col])
                # tag_pos = np.array([100+31.85*np.cos(interval[i]),100+31.85*np.sin(interval[i])])
                true_dist = np.linalg.norm(anchor_pos - tag_pos, axis = 1) #計算Tag到各Anchor的距離
                flag = time.time()
                # for j in range(itr):
                #     z.append(true_dist + np.random.normal(0, 0.1, 4))
                # true_dist = Kalman(z,itr)[0]
                true_dist += (np.random.normal(0, 0.1, 4)) #加入誤差
                true_dist_diff = true_dist - true_dist[0] #計算距離差(相對於Center)
                # last_pos = estimated_tag_pos #Gradient Descent的起點，注意不能為(0, 0)
                # last_pos = np.array([np.sum(anchor_pos[:,0])/4,np.sum(anchor_pos[:,1])/4])
                last_pos = np.array([9*random.random(),5*random.random()])
                h = 0 #高度補償
                
                estimated_tag_pos = multiAnchorPositioning(anchor_pos, true_dist, last_pos, h)
                t = time.time() - flag
                est += (estimated_tag_pos[0]-tag_pos[0])**2+(estimated_tag_pos[1]-tag_pos[1])**2
                spend_time+=t
                # print("The %d times:"%(i+1),t,"s")
            est = np.sqrt(est/times)
            print("RMS error: ",est,"m")
            print("average time",spend_time/times,"s")
            #     plt.plot(alpha,spend_time/times,"bo")
            # plt.show()