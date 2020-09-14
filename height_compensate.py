import numpy as np
import math
import time
import matplotlib.pyplot as plt
import random

def multiAnchorPositioning(anchor_list, distance, last_pos, h , alpha=1):
    height = np.ones((4,1))*h
    anchor_list = np.hstack((anchor_list,height))
    x = last_pos[0]
    y = last_pos[1]
    H = np.ones((len(anchor_list)-1,1))
    G = np.ones((len(anchor_list)-1,2))
    tag = np.array([x,y,0])
    G1 = np.ones((2,len(anchor_list)-1))
    W = np.ones((2,2))
    Q = np.identity(len(anchor_list)-1)
    Q_inv = np.linalg.inv(Q)
    R = np.ones((len(anchor_list)-1,1))
    temp = np.ones(R.shape)
    flag = 0


    #-----------------------------extended LS------------------------
    L = np.zeros((6,5))
    z = np.zeros((6,1))
    row = 0
    for i in range(len(anchor_list)-1):
        for j in range(i,len(anchor_list)-1):
            L[row,0:2] = np.array([anchor_list[j+1][0]-anchor_list[0][0],anchor_list[j+1][1]-anchor_list[0][1]])
            L[row,i+2] = distance[j+1]-distance[i]
            z[row] = 0.5*np.array([(anchor_list[j+1][0]**2+anchor_list[j+1][1]**2+h**2)-(anchor_list[0][0]**2+anchor_list[0][1]**2+h**2)-(distance[j+1]-distance[0])**2])
            row+=1
    ans = np.dot(np.dot(np.linalg.inv(np.dot(L.T,L)),L.T),z)
    tag = np.array([ans[0][0],ans[1][0],0])
    # ---------------------------------------------------------------
    const = 16.65*10^-12*299792458
    # ----------------------------------two stage Taylor-----------------------------
    for times in range(100):
        for i in range(len(anchor_list)-1):
            H[i] = np.array([distance[i+1]-distance[0]-(np.linalg.norm(anchor_list[i+1]-tag)-np.linalg.norm(anchor_list[0]-tag))])
            partial_x = (anchor_list[0,0]-tag[0])/(np.linalg.norm(anchor_list[0]-tag))-(anchor_list[i+1,0]-tag[0])/(np.linalg.norm(anchor_list[i+1]-tag))
            partial_y = (anchor_list[0,1]-tag[1])/(np.linalg.norm(anchor_list[0]-tag))-(anchor_list[i+1,1]-tag[1])/(np.linalg.norm(anchor_list[i+1]-tag))
            G[i] = np.array([partial_x,partial_y])
            if partial_x==0 or partial_y==0:
                return tag
            temp[i] = np.array([np.linalg.norm(anchor_list[i+1]-tag)-np.linalg.norm(anchor_list[0]-tag)])
        if flag<10:
            R = np.copy(temp)
        else:
            R = np.hstack((R,temp))  
            for j in range(len(anchor_list)-1):
                Q[j,j] = np.std(R[j])
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
            tag[0:2] = tag[0:2] + delta.T[0]
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
    anchor_pos = np.array([[0, 0], [0, 7.42], [12.985, 7.436], [12.985, 0]])
    tag_pos = np.array([[3.047,4.206], [3.046,7.318], [3.008,10.784], [6.321,11.779]])
    real_tag_pos = np.array([[4.206,7.42-3.047], [7.318,7.42-3.046], [10.784,7.42-3.008], [11.779,7.42-6.321]])
    f = open("head\\record_"+str(tag[0,0])+"_"+str(tag[0,1])+".txt", 'r')
    raw = f.readlines()
    for i in range(len(raw)):
        data = raw[i].split(',')
        #print(data)
        dist_diff = np.array([0, float(data[7]), float(data[5]), float(data[3])])
        dist_diff *= 15.650040064103e-12
        dist_diff *= 299792458
        Q = np.identity(len(anchor_pos)-1)