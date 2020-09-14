import numpy as np
import math
import time
import matplotlib.pyplot as plt
import random



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
        norm_ai2tag = np.linalg.norm(anchor_list[1:] - tag,axis=1)
        norm_a02tag = np.linalg.norm(anchor_list[0] - tag)
        x_diff = anchor_list[0,0]-tag[0]
        y_diff = anchor_list[0,1]-tag[1]
        for i in range(len(anchor_list)-1):
            H[i] = np.array([dist_diff[0,i+1]-(norm_ai2tag[i]-norm_a02tag)])
            partial_x = (x_diff)/(norm_a02tag)-(anchor_list[i+1,0]-tag[0])/(norm_ai2tag[i])
            partial_y = (y_diff)/(norm_a02tag)-(anchor_list[i+1,1]-tag[1])/(norm_ai2tag[i])
            G[i] = np.array([partial_x,partial_y])
            if partial_x==0 or partial_y==0:
                return tag[0:2]
        # a02tag = anchor_list[0] - tag
        # norm_a02tag = np.linalg.norm(a02tag)

        # ai2tag = anchor_list[1:] - tag
        # norm_ai2tag = np.linalg.norm(ai2tag,axis=1)
        # H = np.array([(dist_diff[0,1:] - (norm_ai2tag-norm_a02tag))]).T
        # G = (a02tag/norm_a02tag - ai2tag/norm_ai2tag)[:,0:2]
        # if np.sum(G==0)!=0:
        #     return tag[0:2]
        delta = np.dot(np.dot(np.dot(np.linalg.pinv(np.dot(G.T,np.dot(Q_inv,G))),G.T),Q_inv),H)
        # delta = np.dot(np.dot(np.linalg.inv(np.dot(G.T,G)),G.T),H)
        for j in range(len(anchor_list)-1):
            partial_xx = 1/norm_ai2tag[j] * (1-((anchor_list[j+1,0]-tag[0])/(norm_ai2tag[j]))**2) - 1/norm_a02tag * (1-(x_diff/(norm_a02tag))**2)
            partial_xy = -1/norm_ai2tag[j] * (anchor_list[j+1,0]-tag[0])/norm_ai2tag[j] * (anchor_list[j+1,1]-tag[1])/norm_ai2tag[j] + 1/norm_a02tag * (x_diff)/norm_a02tag * (y_diff)/norm_a02tag
            partial_yy = -1/norm_a02tag * (1-(y_diff/(norm_a02tag))**2) + 1/norm_ai2tag[j] * (1-((anchor_list[j+1,1]-tag[1])/(norm_ai2tag[j]))**2)
            W = np.array([[partial_xx,partial_xy],[partial_xy,partial_yy]])
            G1[:,j] = np.dot(W.T,delta)[:,0]
        G = G + 0.5*G1.T
        delta = np.dot(np.dot(np.dot(np.linalg.pinv(np.dot(G.T,np.dot(Q_inv,G))),G.T),Q_inv),H).T
        
        if np.sqrt(delta[0,0]**2+delta[0,1]**2)>1e+2:
            pass
        else:
            tag[0:2] = tag[0:2] + delta[0]
        if np.linalg.norm(delta[0])<0.001:
            break
    return tag[0:2]

def Kalman(z,itr):
    X = np.zeros((1,4))
    P = 1
    R = 0.1
    for i in range(itr):
        K = P/(P+R)
        X = X + K*(z[i]-X)
        P = (1-K)*P
    return X

def GD(anchor_list, distance, last_pos, h):
    times = 200
    l_rate_x = 8
    l_rate_y = 8
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



#----- Example -----
if __name__ == '__main__':
    anchor_pos = np.array([[0, 0], [0, 7.42], [12.985, 7.436], [12.985, 0]])
    tag_pos = np.array([[3.047,4.206], [3.046,7.318], [3.008,10.784], [6.321,11.779]])
    real_tag_pos = np.array([[4.206,7.42-3.047], [7.318,7.42-3.046], [10.784,7.42-3.008], [11.779,7.42-6.321]])
    for test_sample in range(len(tag_pos)):
        dist_diff_array = np.zeros((1,len(anchor_pos)-1))
        pos_record_all = []
        RMSE = 0
        total_time = 0
        f = open("record_"+str(tag_pos[test_sample,0])+"_"+str(tag_pos[test_sample,1])+".txt", 'r')
        raw = f.readlines()
        for i in range(len(raw)):
            flag = time.time()
            data = raw[i].split(',')
            #print(data)
            dist_diff = np.array([[0, float(data[7]), float(data[5]), float(data[3])]])
            dist_diff *= 15.650040064103e-12
            dist_diff *= 299792458
            if i < 10:
                dist_diff_array = np.copy(dist_diff[0,1:])
                Q = np.identity(len(anchor_pos)-1)
            else:
                dist_diff_array = np.vstack((dist_diff_array,dist_diff[0,1:]))
                for j in range(len(anchor_pos)-1):
                    Q[j,j] = np.std(dist_diff_array[:,j])
            
            estimated_pos = Taylor(anchor_pos,dist_diff,0,Q)
            # estimated_pos = GD(anchor_pos,dist_diff[0],[1e-6,1e-6],0)
            pos_record_all.append(estimated_pos)
            t = time.time() - flag
            err = np.linalg.norm(estimated_pos-real_tag_pos[test_sample])
            # print("Test for",i+1,"times....")
            # print("spend time: ",t,"s")
            # print("error:      ",err,"m")
            RMSE += np.sum((estimated_pos-real_tag_pos[test_sample])**2)
            total_time += t
        print("------------------------------------------------------")
        print("tag pos:",real_tag_pos[test_sample])
        print("average spend time: ",total_time/len(raw)*1000,"ms")
        print("RMS error:          ",np.sqrt(RMSE/len(raw))*100,"cm")
    #     pos_record_all = np.array(pos_record_all)
    #     plt.scatter(x = pos_record_all[:,0], y = pos_record_all[:,1], color = 'red', marker = 'x', s = 10)
    # plt.scatter(x = real_tag_pos[:,0], y = real_tag_pos[:,1], color = 'orange', marker = 's', s = 50, label = 'Tag')
    # plt.scatter(x = anchor_pos[:,0], y = anchor_pos[:,1], color = 'black', marker = '^', s = 50, label = 'Anchor')
    # plt.xlim(-1, 16)
    # plt.ylim(-1, 9)
    # plt.legend(loc='upper right')
    # plt.title("Scatter plot of calculated points")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.grid()
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.show()