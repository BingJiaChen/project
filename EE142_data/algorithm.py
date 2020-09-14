import numpy as np
from math import *
from numpy.linalg import *

def GD(anchor_list, distance, last_pos, h):
    times = 200
    l_rate_x = 5
    l_rate_y = 5
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
    for i in range(times):
        f = np.sqrt((x - x_p1) ** 2 + (y - y_p1) ** 2 + h**2)-np.sqrt((x - x_p2) ** 2 + (y - y_p2) ** 2 + h**2)-r_p
        f_partial_x = 2*np.sum(f*((x - x_p1)/np.sqrt((x - x_p1)** 2 + (y - y_p1)** 2 + h**2) - (x - x_p2)/np.sqrt((x - x_p2)** 2 + (y - y_p2)** 2 + h**2)), axis = 0)
        f_partial_y = 2*np.sum(f*((y - y_p1)/np.sqrt((x - x_p1)** 2 + (y - y_p1)** 2 + h**2) - (y - y_p2)/np.sqrt((x - x_p2)** 2 + (y - y_p2)** 2 + h**2)), axis = 0)
        wX += f_partial_x ** 2
        ada1 = np.sqrt(wX)
        wY += f_partial_y ** 2
        ada2 = np.sqrt(wY)

        x = x - l_rate_x * f_partial_x / ada1
        y = y - l_rate_y * f_partial_y / ada2
        #print(i,x,y)
    Ans = np.array([x[0], y[0]])
    return Ans

def LS(anchor_list, dist_diff, h):
    L = np.zeros((6,5))
    z = np.zeros((6,1))
    row = 0
    for i in range(len(anchor_list)-1):
        for j in range(i,len(anchor_list)-1):
            L[row,0:2] = np.array([anchor_list[j+1][0]-anchor_list[0][0],anchor_list[j+1][1]-anchor_list[0][1]])
            L[row,i+2] = dist_diff[0,j+1] - dist_diff[0,i]
            z[row] = 0.5*np.array([(anchor_list[j+1][0]**2+anchor_list[j+1][1]**2+h**2)-(anchor_list[0][0]**2+anchor_list[0][1]**2+h**2)-(dist_diff[0,i+1])**2])
            row+=1
    ans = np.dot(np.dot(np.linalg.pinv(np.dot(L.T,L)),L.T),z)
    tag = np.array([ans[0][0],ans[1][0],0])
    return tag[0:2]



def Chan(anchor_pos, dist_diff, Q):
    anchor_num = len(anchor_pos)
    k = (anchor_pos**2).sum(1)
    h = 0.5* (dist_diff**2 - k[1:anchor_num] + k[0])
    Ga = []
    for i in range(1, anchor_num):
        Ga.append([anchor_pos[i][0]-anchor_pos[0][0], anchor_pos[i][1]-anchor_pos[0][1], dist_diff[i-1]])
    Ga = np.array(Ga)
    Ga = -Ga

    #print(Ga)
    #print(Q)
    #print(h)
    Za = pinv((Ga.T).dot(pinv(Q)).dot(Ga)).dot((Ga.T).dot(pinv(Q)).dot(h))

    real_dis = np.sqrt(((anchor_pos[1:anchor_num]-Za[0:2])**2).sum(1))
    Ba = np.diag(real_dis)
    Fa = Ba.dot(Q).dot(Ba)
    Zacov = pinv((Ga.T).dot(pinv(Fa)).dot(Ga))

    Za1 = pinv((Ga.T).dot(pinv(Fa)).dot(Ga)).dot((Ga.T)).dot(pinv(Fa)).dot(h)
    real_dis1 = np.sqrt(((anchor_pos[1:anchor_num]-Za1[0:2])**2).sum(1))
    Ba1 = np.diag(real_dis1)
    Fa1 = Ba1.dot(Q).dot(Ba)
    Zacov1 = pinv((Ga.T).dot(pinv(Fa1)).dot(Ga))

    Ga1 = np.array([[1,0], [0,1], [1,1]])
    h1 = np.array([(Za1[0]-anchor_pos[0][0])**2, (Za1[1]-anchor_pos[0][1])**2, Za1[2]**2])
    Bb = np.diag([Za1[0]-anchor_pos[0][0], Za1[1]-anchor_pos[0][1], Za1[2]])
    Fa2 = 4* (Bb).dot(Zacov1).dot(Bb)
    Za2 = pinv((Ga1.T).dot(pinv(Fa2)).dot(Ga1)).dot((Ga1.T)).dot(pinv(Fa2)).dot(h1)
    pos1 = np.sqrt(Za2) + anchor_pos[0]
    pos2 = -np.sqrt(Za2) + anchor_pos[0]
    pos3 = [np.sqrt(Za2[0]), -np.sqrt(Za2[1])] + anchor_pos[0]
    pos4 = [-np.sqrt(Za2[0]), -np.sqrt(Za2[1])] + anchor_pos[0]
    pos = [pos1, pos2, pos3, pos4]
    return pos

def Taylor(anchor_list, dist_diff, h, Q ):
    height = np.ones((4,1))*h
    anchor_list = np.hstack((anchor_list,height))
    H = np.ones((len(anchor_list)-1,1))
    G = np.ones((len(anchor_list)-1,2))
    G1 = np.ones((2,len(anchor_list)-1))
    W = np.ones((2,2))
    Q_inv = np.linalg.inv(Q)


    #-----------------------------extended LS------------------------
    L = np.zeros((6,5))
    z = np.zeros((6,1))
    row = 0
    for i in range(len(anchor_list)-1):
        for j in range(i,len(anchor_list)-1):
            L[row,0:2] = np.array([anchor_list[j+1][0]-anchor_list[0][0],anchor_list[j+1][1]-anchor_list[0][1]])
            L[row,i+2] = dist_diff[0,j+1] - dist_diff[0,i]
            z[row] = 0.5*np.array([(anchor_list[j+1][0]**2+anchor_list[j+1][1]**2+h**2)-(anchor_list[0][0]**2+anchor_list[0][1]**2+h**2)-(dist_diff[0,i+1])**2])
            row+=1
    ans = np.dot(np.dot(np.linalg.pinv(np.dot(L.T,L)),L.T),z)
    tag = np.array([ans[0][0],ans[1][0],0])
    
    # ---------------------------------------------------------------

    # ----------------------------------two stage Taylor-----------------------------
    for times in range(100):
        for i in range(len(anchor_list)-1):
            H[i] = np.array([dist_diff[0,i+1]-(np.linalg.norm(anchor_list[i+1]-tag)-np.linalg.norm(anchor_list[0]-tag))])
            partial_x = (anchor_list[0,0]-tag[0])/(np.linalg.norm(anchor_list[0]-tag))-(anchor_list[i+1,0]-tag[0])/(np.linalg.norm(anchor_list[i+1]-tag))
            partial_y = (anchor_list[0,1]-tag[1])/(np.linalg.norm(anchor_list[0]-tag))-(anchor_list[i+1,1]-tag[1])/(np.linalg.norm(anchor_list[i+1]-tag))
            G[i] = np.array([partial_x,partial_y])
            if partial_x==0 or partial_y==0:
                return tag[0:2]
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
        if np.linalg.norm(delta.T[0])<0.001:
            break

    return tag[0:2]


def Taylor_GD(anchor_pos, dist_diff, init_pos, Q):
    count = 0
    tag_pos = init_pos.copy()
    distance = dist_diff.copy()
    x = tag_pos[0]
    y = tag_pos[1]
    l_rate_x = 1
    l_rate_y = 1
    l_rate = 1
    center_pos = anchor_pos[0]
    l = int(len(anchor_pos)*(len(anchor_pos)-1)/2)
    x_p1 = np.zeros((l, 1))
    y_p1 = np.zeros((l, 1))
    x_p2 = np.zeros((l, 1))
    y_p2 = np.zeros((l, 1))
    r_p = np.zeros((l, 1))
    counter = 0
    for i in range(len(anchor_pos)-1):
        for j in np.arange(i+1, len(anchor_pos)):
            x_p1[counter] = (anchor_pos[i][0])
            y_p1[counter] = (anchor_pos[i][1])
            x_p2[counter] = (anchor_pos[j][0])
            y_p2[counter] = (anchor_pos[j][1])
            r_p[counter] = (distance[i]-distance[j])
            counter = counter + 1

    WX = 1e-10
    WY = 1e-10
    W = 1e-10
    while True:
        count += 1
        h = []
        G = []
        R1 = np.linalg.norm(tag_pos - center_pos)
        for i in range(len(anchor_pos)):
            if i > 0:
	            Ri = np.linalg.norm(tag_pos - anchor_pos[i])
	            h.append(dist_diff[i]-(Ri-R1))
	            G.append([(center_pos[0]-tag_pos[0])/R1-(anchor_pos[i][0]-tag_pos[0])/Ri,
	                      (center_pos[1]-tag_pos[1])/R1-(anchor_pos[i][1]-tag_pos[1])/Ri ])
        h = np.array(h)
        G = np.array(G)
        f = np.sqrt((x - x_p1) ** 2 + (y - y_p1) ** 2)-np.sqrt((x - x_p2) ** 2 + (y - y_p2) ** 2)-r_p
        f_partial_x = 2*np.sum(f*((x - x_p1)/np.sqrt((x - x_p1)** 2 + (y - y_p1)** 2) - (x - x_p2)/np.sqrt((x - x_p2)** 2 + (y - y_p2)** 2)), axis = 0)
        f_partial_y = 2*np.sum(f*((y - y_p1)/np.sqrt((x - x_p1)** 2 + (y - y_p1)** 2) - (y - y_p2)/np.sqrt((x - x_p2)** 2 + (y - y_p2)** 2)), axis = 0)
        #WX += f_partial_x[0]**2
        #WY += f_partial_y[0]**2
        
        #print('GD:',[-l_rate_x*f_partial_x[0]/np.sqrt(WX), -l_rate_y*f_partial_y[0]/np.sqrt(WY)])
        delta_pos_Taylor = pinv(G.T @ pinv(Q) @ G) @ G.T @ pinv(Q) @ h
        delta_pos_GD = np.array([-l_rate_x*f_partial_x[0], -l_rate_y*f_partial_y[0]])
        #delta_pos = pinv(G.T @ G) @ G.T @ h
       	delta_pos = delta_pos_Taylor + delta_pos_GD
       	#if count < 4:
        #	print('Tag_pos:',tag_pos, 'Taylor:',delta_pos_Taylor, 'GD:', delta_pos_GD)
        '''WX += (delta_pos[0]**2)
        WY += (delta_pos[1]**2)'''
        W += np.sum(delta_pos**2)
        '''delta_X = l_rate_x*delta_pos[0]/np.sqrt(WX)
        delta_Y = l_rate_y*delta_pos[1]/np.sqrt(WY)
        delta_pos = np.array([delta_X,delta_Y])'''
        delta_pos = l_rate*delta_pos/np.sqrt(W)
        tag_pos += delta_pos
        #print(tag_pos)
        x = tag_pos[0]
        y = tag_pos[1]
        if np.linalg.norm(delta_pos) < 0.001 or count > 100:
            break
    #print(count)
    return tag_pos




def Taylor2(anchor_pos, dist_diff, init_pos):
    count = 0
    tag_pos = init_pos
    #center_pos = anchor_pos[0]
    #anchor_pos = anchor_pos[1:4]
    A = np.ones((len(anchor_pos)-1, 2))
    B = np.ones((len(anchor_pos)-1, 1))
    while count < 200:
        count += 1
        for i in range(len(anchor_pos)-1):
            A[i][0] = tag_pos[0] - (anchor_pos[i+1][0] - anchor_pos[0][0])
            A[i][1] = tag_pos[1] - (anchor_pos[i+1][1] - anchor_pos[0][1])
            
            B[i] = 0.5* ((dist_diff[i+1])**2-A[i][0]**2-A[i][1]**2) + A[i][0]*tag_pos[0] + A[i][1]*tag_pos[1]
        
        #print(A, B)
        delta_pos = np.linalg.pinv(A.T @ A) @ A.T @ B
        tag_pos = delta_pos.reshape((1,2))[0]
        #print(tag_pos)
        #tag_pos += delta_pos
        #if np.linalg.norm(delta_pos) < 0.0001:
         #   break
    #print(count)
    return tag_pos