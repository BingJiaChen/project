import numpy as np
import math
import time
import matplotlib.pyplot as plt
import random

def multiAnchorPositioning(anchor_list, distance, last_pos, h):
	l_rate_x = 10
	l_rate_y = 10	
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
	flag =True
	while True:
		x_old = x
		y_old = y
		f = np.sqrt((x - x_p1) ** 2 + (y - y_p1) ** 2 + h**2)-np.sqrt((x - x_p2) ** 2 + (y - y_p2) ** 2 + h**2)-r_p
		# print(f)
		f_partial_x = 2*np.sum(f*((x - x_p1)/np.sqrt((x - x_p1)** 2 + (y - y_p1)** 2 + h**2) - (x - x_p2)/np.sqrt((x - x_p2)** 2 + (y - y_p2)** 2 + h**2)), axis = 0)[0]
		f_partial_y = 2*np.sum(f*((y - y_p1)/np.sqrt((x - x_p1)** 2 + (y - y_p1)** 2 + h**2) - (y - y_p2)/np.sqrt((x - x_p2)** 2 + (y - y_p2)** 2 + h**2)), axis = 0)[0]
		wX += f_partial_x ** 2
		ada1 = np.sqrt(wX)
		wY += f_partial_y ** 2
		ada2 = np.sqrt(wY)
		# if flag:
		# 	print(f_partial_x,f_partial_y)
		# 	flag = False
		x = x - l_rate_x * f_partial_x / ada1
		y = y - l_rate_y * f_partial_y / ada2
		#print(i,x,y)
		est = np.sqrt((x-x_old)**2+(y-y_old)**2)
		if est < 0.001:
			break
	Ans = np.array([x,y])
	# print(Ans)
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
							[9, 0],  #Anchor1
							[9, 5],  #Anchor2
							[0, 5]]) #Anchor3
		# tag_pos = np.array([4+5*np.cos(interval[i]),3+5*np.sin(interval[i])])
		tag_pos = np.array([-10,0])
		# tag_pos = np.array([100+31.85*np.cos(interval[i]),100+31.85*np.sin(interval[i])])
		true_dist = np.linalg.norm(anchor_pos - tag_pos, axis = 1) #計算Tag到各Anchor的距離
		true_dist += (np.random.normal(0, 0.1, 4)) #加入誤差
		true_dist_diff = true_dist - true_dist[0] #計算距離差(相對於Center)
		# last_pos = estimated_tag_pos #Gradient Descent的起點，注意不能為(0, 0)
		last_pos = np.array([9*random.random(),5*random.random()])
		h = 0 #高度補償
		flag = time.time()
		estimated_tag_pos = multiAnchorPositioning(anchor_pos, true_dist, last_pos, h)
		t = time.time() - flag
		# plt.plot(estimated_tag_pos[0],estimated_tag_pos[1],"bo")
		est += (estimated_tag_pos[0]-tag_pos[0])**2+(estimated_tag_pos[1]-tag_pos[1])**2
		spend_time+=t
		# print("The %d times:"%(i+1),t,"s")
	est = np.sqrt(est/times)
	print("RMS error: ",est,"m")
	print("average time",spend_time/times,"s")