import numpy as np
import math
import time
import matplotlib.pyplot as plt
import random

def multiAnchorPositioning(anchor_list, distance, last_pos, h , alpha=1):
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
    H = np.ones((len(anchor_list)-1,1))
    G = np.ones((len(anchor_list)-1,2))
    tag = np.array([x,y])
    G1 = np.ones((2,len(anchor_list)-1))
    W = np.ones((2,2))
    Q = np.identity(len(anchor_list)-1)
    Q_inv = np.linalg.inv(Q)
    R = np.ones((len(anchor_list)-1,1))
    temp = np.ones(R.shape)
    flag = 0
    alpha = 10
    do_GD = False

    x = tag[0]
    y = tag[1]
    wX = 0
    wY = 0
    Dx = 0
    Dy = 0
    i = 1
    eps = 1e-8
    beta1=0.9
    beta2=0.999 
    beta = 0.9
    alpha=10
    #-----------------------------LS------------------------
    L = np.zeros((6,5))
    z = np.zeros((6,1))
    row = 0
    for i in range(len(anchor_list)-1):
        for j in range(i,len(anchor_list)-1):
            L[row,0:2] = np.array([anchor_list[j+1][0]-anchor_list[0][0],anchor_list[j+1][1]-anchor_list[0][1]])
            L[row,i+2] = distance[i+1]-distance[0]
            z[row] = 0.5*np.array([(anchor_list[j+1][0]**2+anchor_list[j+1][1]**2)-(anchor_list[0][0]**2+anchor_list[0][1]**2)-(distance[j+1]-distance[0])**2])
            row+=1
    ans = np.dot(np.dot(np.linalg.inv(np.dot(L.T,L)),L.T),z)
    tag = np.array([ans[0][0],ans[1][0]])
    # ---------------------------------------------------------
    for zzz in range(100):
        x_old = x
        y_old = y
        for i in range(len(anchor_list)-1):
            H[i] = np.array([distance[i+1]-distance[0]-(np.linalg.norm(anchor_list[i+1]-tag)-np.linalg.norm(anchor_list[0]-tag))])
            x1 = (anchor_list[0,0]-tag[0])/(np.linalg.norm(anchor_list[0]-tag))-(anchor_list[i+1,0]-tag[0])/(np.linalg.norm(anchor_list[i+1]-tag))
            x2 = (anchor_list[0,1]-tag[1])/(np.linalg.norm(anchor_list[0]-tag))-(anchor_list[i+1,1]-tag[1])/(np.linalg.norm(anchor_list[i+1]-tag))
            G[i] = np.array([x1,x2])
            # if x1==0 or x2==0:
            #     return tag
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
        # for j in range(len(anchor_list)-1):
        #     partial_xx = 1/np.linalg.norm(anchor_list[j+1]-tag) * (1-((anchor_list[j+1,0]-tag[0])/(np.linalg.norm(anchor_list[j+1]-tag)))**2) - 1/np.linalg.norm(anchor_list[0]-tag) * (1-((anchor_list[0,0]-tag[0])/(np.linalg.norm(anchor_list[0]-tag)))**2)
        #     partial_xy = -1/np.linalg.norm(anchor_list[j+1]-tag) * (anchor_list[j+1,0]-tag[0])/np.linalg.norm(anchor_list[j+1]-tag) * (anchor_list[j+1,1]-tag[1])/np.linalg.norm(anchor_list[j+1]-tag) + 1/np.linalg.norm(anchor_list[0]-tag) * (anchor_list[0,0]-tag[0])/np.linalg.norm(anchor_list[0]-tag) * (anchor_list[0,1]-tag[1])/np.linalg.norm(anchor_list[0]-tag)
        #     partial_yy = -1/np.linalg.norm(anchor_list[0]-tag) * (1-((anchor_list[0,1]-tag[1])/(np.linalg.norm(anchor_list[0]-tag)))**2) + 1/np.linalg.norm(anchor_list[j+1]-tag) * (1-((anchor_list[j+1,1]-tag[1])/(np.linalg.norm(anchor_list[j+1]-tag)))**2)
        #     W = np.array([[partial_xx,partial_xy],[partial_xy,partial_yy]])
        #     G1[:,j] = np.dot(W.T,delta)[:,0]
        # G = G + 0.5*G1.T
        # delta = np.dot(np.dot(np.dot(np.linalg.pinv(np.dot(G.T,np.dot(Q_inv,G))),G.T),Q_inv),H)
        if np.sqrt(delta.T[0,0]**2+delta.T[0,1]**2)>1e+2:
            delta = np.zeros((2,1))
        f = np.sqrt((x - x_p1) ** 2 + (y - y_p1) ** 2 + h**2)-np.sqrt((x - x_p2) ** 2 + (y - y_p2) ** 2 + h**2)-r_p
        f_partial_x = 2*np.sum(f*((x - x_p1)/np.sqrt((x - x_p1)** 2 + (y - y_p1)** 2 + h**2) - (x - x_p2)/np.sqrt((x - x_p2)** 2 + (y - y_p2)** 2 + h**2)), axis = 0)[0]
        f_partial_y = 2*np.sum(f*((y - y_p1)/np.sqrt((x - x_p1)** 2 + (y - y_p1)** 2 + h**2) - (y - y_p2)/np.sqrt((x - x_p2)** 2 + (y - y_p2)** 2 + h**2)), axis = 0)[0]
        # wX = beta*wX + (1-beta)*(f_partial_x ** 2 + delta.T[0,0]**2)
        wX += (f_partial_x ** 2 + delta.T[0,0]**2)
        ada1 = np.sqrt(wX)
        # wY = beta*wY + (1-beta)*(f_partial_y ** 2 + delta.T[0,1]**2)
        wY += (f_partial_y ** 2 + delta.T[0,1]**2)
        ada2 = np.sqrt(wY)

        x = x + (- l_rate_x * f_partial_x + delta.T[0,0])/ ada1
        y = y + (- l_rate_y * f_partial_y + delta.T[0,1])/ ada2
        est = np.sqrt((x-x_old)**2+(y-y_old)**2)
        tag = np.array([x,y])
        if est<=0.001:
            break
    tag = np.array([x,y])
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
    x = []
    y = []
    color = []
    est=0
    spend_time=0
    interval = np.linspace(0,2*np.pi,100)
    estimated_tag_pos = np.array([1e-6,1e-6])
    times = 100
    itr = 100
    alpha = 1.9
    for row in range(19):
        for col in range(11):
            est = 0
            spend_time = 0
            print("Test tag position: ",([0.5*row,0.5*col]))
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
                tag_pos = np.array( [0.5*row,0.5*col])
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
            x.append(tag_pos[0])
            y.append(tag_pos[1])
            if est>5:
                est = 5
            color.append(est)
            
            print("average time",spend_time/times,"s")
            #     plt.plot(alpha,spend_time/times,"bo")
            # plt.show()

    plt.scatter(x,y,c=color,cmap='viridis')
    plt.plot([0,9],[0,0],ls='--',c='r')
    plt.plot([9,9],[0,5],ls='--',c='r')
    plt.plot([9,0],[5,5],ls='--',c='r')
    plt.plot([0,0],[5,0],ls='--',c='r')
    plt.colorbar()
    plt.show()