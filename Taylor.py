import numpy as np
import math
import time
import matplotlib.pyplot as plt
import random

def multiAnchorPositioning(anchor_list, distance, last_pos, h , alpha):
    # l_rate_x = 1
    # l_rate_y = 1	
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
    H = np.ones((len(anchor_list)-1,1))
    G = np.ones((len(anchor_list)-1,2))
    tag = np.array([x,y])
    # Q_t = 0.5*np.ones((len(anchor_list)-1,len(anchor_list)-1))
    # Q = 0.5*np.identity(len(anchor_list)-1)
    # Q = Q + Q_t
    Q = np.identity(len(anchor_list)-1)
    Q_inv = np.linalg.inv(Q)
    R = np.ones((len(anchor_list)-1,1))
    temp = np.ones(R.shape)
    flag = 0


    #-----------------------------LS------------------------
    # L = np.zeros((6,5))
    # z = np.zeros((6,1))
    # row = 0
    # for i in range(len(anchor_list)-1):
    #     for j in range(i,len(anchor_list)-1):
    #         L[row,0:2] = np.array([anchor_list[j+1][0]-anchor_list[0][0],anchor_list[j+1][1]-anchor_list[0][1]])
    #         L[row,i+2] = distance[i+1]-distance[0]
    #         z[row] = 0.5*np.array([(anchor_list[j+1][0]**2+anchor_list[j+1][1]**2)-(anchor_list[0][0]**2+anchor_list[0][1]**2)-(distance[j+1]-distance[0])**2])
    #         row+=1
    # ans = np.dot(np.dot(np.linalg.inv(np.dot(L.T,L)),L.T),z)
    # tag = np.array([ans[0][0],ans[1][0]])
    # ---------------------------------------------------------

    for times in range(100):
        for i in range(len(anchor_list)-1):
            H[i] = np.array([distance[i+1]-distance[0]-(np.linalg.norm(anchor_list[i+1]-tag)-np.linalg.norm(anchor_list[0]-tag))])
            x1 = (anchor_list[0,0]-tag[0])/(np.linalg.norm(anchor_list[0]-tag))-(anchor_list[i+1,0]-tag[0])/(np.linalg.norm(anchor_list[i+1]-tag))
            x2 = (anchor_list[0,1]-tag[1])/(np.linalg.norm(anchor_list[0]-tag))-(anchor_list[i+1,1]-tag[1])/(np.linalg.norm(anchor_list[i+1]-tag))
            G[i] = np.array([x1,x2])
            if x1==0 or x2==0:
                return tag
        #     temp[i] = np.array([np.linalg.norm(anchor_list[i+1]-tag)-np.linalg.norm(anchor_list[0]-tag)])
        # if flag<10:
        #     R = np.copy(temp)
        # else:
        #     R = np.hstack((R,temp))  
        #     for j in range(len(anchor_list)-1):
        #         Q[j,j] = np.std(R[j])
        #     # Q = np.cov(R)
        #     Q_inv = np.linalg.pinv(Q)
        # delta = np.dot(np.dot(np.dot(np.linalg.pinv(np.dot(G.T,np.dot(Q_inv,G))),G.T),Q_inv),H)
        delta = np.dot(np.dot(np.linalg.inv(np.dot(G.T,G)),G.T),H)
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

def Kalman(z,itr):
    X = np.zeros((1,4))
    P = 1
    R = 0.1
    for i in range(itr):
        K = P/(P+R)
        X = X + K*(z[i]-X)
        P = (1-K)*P
    return X


#----- Example -----
if __name__ == '__main__':
    est=0
    spend_time=0
    interval = np.linspace(0,2*np.pi,100)
    estimated_tag_pos = np.array([100,100])
    times = 100
    itr = 500
    #[0, 0], [9.134, 0], [9.104, 5.187], [0, 5.182]
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
        tag_pos = np.array([3.86,2.22])
        # tag_pos = np.array([100+150*np.cos(interval[i]),100+150*np.sin(interval[i])])
        true_dist = np.linalg.norm(anchor_pos - tag_pos, axis = 1) #計算Tag到各Anchor的距離
        # for j in range(itr):
        #     z.append(true_dist + np.random.normal(0, 0.1, 4))
        # true_dist = Kalman(z,itr)[0]
        true_dist += (np.random.normal(0, 0.1, 4)) #加入誤差
        true_dist_diff = true_dist - true_dist[0] #計算距離差(相對於Center)
        # last_pos = estimated_tag_pos #Gradient Descent的起點，注意不能為(0, 0)
        # last_pos = np.array([np.sum(anchor_pos[:,0])/4,np.sum(anchor_pos[:,1])/4])
        last_pos = np.array([9*random.random(),5*random.random()])
        h = 0 #高度補償
        flag = time.time()
        estimated_tag_pos = multiAnchorPositioning(anchor_pos, true_dist, last_pos, h,1)
        t = time.time() - flag
        # plt.plot(estimated_tag_pos[0],estimated_tag_pos[1],"bo")
        est += (estimated_tag_pos[0]-tag_pos[0])**2+(estimated_tag_pos[1]-tag_pos[1])**2
        spend_time+=t
        # print("The %d times:"%(i+1),t,"s")
    est = np.sqrt(est/times)
    print("RMS error: ",est,"m")
    print("average time",spend_time/times,"s")