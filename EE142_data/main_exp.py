import os
import sys
import traceback
import time
import numpy as np
from algorithm import *
import matplotlib.pyplot as plt


anchor_pos = np.array([[7.42, 0], [7.436, 12.985], [0, 12.985], [0, 0]])
anchor_pos = np.array([[0, 0], [0, 7.42], [12.985, 7.436], [12.985, 0]])

#tag_pos = np.array([[1, 1], [6, 2], [11, 3], [3, 4], [8, 5], [13, 6]])#
tag_pos = np.array([[3.047,4.206], [3.046,7.318], [3.008,10.784], [6.321,11.779]])
real_tag_pos = np.array([[4.206,7.42-3.047], [7.318,7.42-3.046], [10.784,7.42-3.008], [11.779,7.42-6.321]])
height = 0

def main():
    try:
        pos_record_all = []
        counter = 0
        for tag in tag_pos:
            pos_record = []
            var = []
            counter += 1
            f = open("record_"+str(tag[0])+"_"+str(tag[1])+".txt", 'r')
            raw = f.readlines()
            for i in range(len(raw)):
                data = raw[i].split(',')
                #print(data)
                dist_diff = np.array([[0, float(data[7]), float(data[5]), float(data[3])]])
                dist_diff *= 15.650040064103e-12
                dist_diff *= 299792458
                #print(dist_diff)
                pos_LS = LS(anchor_pos, dist_diff, 0)
                pos_GD = GD(anchor_pos, dist_diff[0], [1e-6, 1e-6], 0)
                if i < 10:
                    dist_diff_array = np.copy(dist_diff[0,1:])
                    Q = np.identity(len(anchor_pos)-1)
                else:
                    dist_diff_array = np.vstack((dist_diff_array,dist_diff[0,1:]))
                    for j in range(len(anchor_pos)-1):
                        Q[j,j] = np.std(dist_diff_array[:,j])
                pos_TG = Taylor_GD(anchor_pos,dist_diff[0],[1e-6,1e-6],Q)
                # pos_Chan = Taylor_GD(anchor_pos, dist_diff[0,1:], Q)
                pos_Taylor = Taylor(anchor_pos, dist_diff, 0, Q)
                #pos_Taylor2 = Taylor2(anchor_pos, dist_diff, [1e-6, 1e-6])
                #print(pos_GD)
                pos_record.append([pos_LS[0:2], pos_GD, pos_TG,pos_Taylor])
                pos_record_all.append([pos_LS[0:2], pos_GD, pos_TG,pos_Taylor])
            pos_record = np.array(pos_record)
            rmse_ls     = np.sqrt(np.nanmean(np.sum((pos_record[:,0]-real_tag_pos[counter-1])**2, axis = 1)))
            rmse_gd     = np.sqrt(np.nanmean(np.sum((pos_record[:,1]-real_tag_pos[counter-1])**2, axis = 1)))
            rmse_chan   = np.sqrt(np.nanmean(np.sum((pos_record[:,2]-real_tag_pos[counter-1])**2, axis = 1)))
            rmse_taylor = np.sqrt(np.nanmean(np.sum((pos_record[:,3]-real_tag_pos[counter-1])**2, axis = 1)))
            print("Tag:", tag, "LS:", rmse_ls, "Chan:", rmse_chan, "GD:", rmse_gd, "Taylor:", rmse_taylor)
            # print("Tag:", tag, "LS:", rmse_ls, "Chan:", rmse_chan, "GD:", rmse_gd)

            plt.subplot(2,2,counter, title = "Tag:"+str(real_tag_pos[counter-1]), xlabel = "RMSE(m)", ylabel = "CDF")
            plt.xlim(0, 5)
            plt.grid()
            err = np.sqrt(np.nansum((pos_record[:,0]-real_tag_pos[counter-1])**2, axis = 1))
            plt.plot(np.sort(err), np.cumsum(err/np.sum(err)), color = 'red', linestyle = '--', label = 'LS')

            err = np.sqrt(np.nansum((pos_record[:,1]-real_tag_pos[counter-1])**2, axis = 1))
            plt.plot(np.sort(err), np.cumsum(err/np.sum(err)), color = 'blue', label = 'GD')

            err = np.sqrt(np.nansum((pos_record[:,2]-real_tag_pos[counter-1])**2, axis = 1))
            plt.plot(np.sort(err), np.cumsum(err/np.sum(err)), color = 'green', linestyle = '-.', label = 'Chan')
            
            err = np.sqrt(np.nansum((pos_record[:,3]-real_tag_pos[counter-1])**2, axis = 1))
            plt.plot(np.sort(err), np.cumsum(err/np.sum(err)), color = 'orange', label = 'Taylor')

            plt.legend(loc='lower right')
        
        plt.show()
        pos_record_all = np.array(pos_record_all)
        #print(pos_record)
        #print(pos_record[:,1,0])
        plt.scatter(x = pos_record_all[:,0,0], y = pos_record_all[:,0,1], color = 'red', marker = 'x', s = 10, label = 'LS')
        plt.scatter(x = pos_record_all[:,2,0], y = pos_record_all[:,2,1], color = 'green', marker = 'v', s = 10, label = 'Chan')
        plt.scatter(x = pos_record_all[:,3,0], y = pos_record_all[:,3,1], color = 'orange', marker = '8', s = 10, label = 'Taylor')
        plt.scatter(x = pos_record_all[:,1,0], y = pos_record_all[:,1,1], color = 'blue', s = 10, label = 'GD')
        plt.scatter(x = real_tag_pos[:,0], y = real_tag_pos[:,1], color = 'orange', marker = 's', s = 50, label = 'Tag')
        plt.scatter(x = anchor_pos[:,0], y = anchor_pos[:,1], color = 'black', marker = '^', s = 50, label = 'Anchor')
        plt.xlim(-1, 16)
        plt.ylim(-1, 9)
        plt.legend(loc='upper right')
        plt.title("Scatter plot of calculated points")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
    except:
        traceback.print_exc()


if __name__ == '__main__':
    main()
