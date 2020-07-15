import numpy as np
import math
import time

def multiAnchorPositioning(anchor_list, distance, last_pos, h):
	times = 200
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
	while True:
		x_old = x
		y_old = y
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
		est = np.sqrt((x-x_old)**2+(y-y_old)**2)
		if est < 0.0001:
			break
	Ans = np.array([x[0], y[0]])
	return Ans


#----- Example -----
if __name__ == '__main__':
	anchor_pos = np.array([[0, 0],  #Acnrho0(Center)
						[500, 0],  #Anchor1
						[500, 500],  #Anchor2
						[0, 500]]) #Anchor3
	tag_pos = np.array([100, 300])
	true_dist = np.linalg.norm(anchor_pos - tag_pos, axis = 1) #計算Tag到各Anchor的距離
	true_dist += np.random.normal(0, 6, 4) #加入誤差
	true_dist_diff = true_dist - true_dist[0] #計算距離差(相對於Center)
	last_pos = [1e-6, 1e-6] #Gradient Descent的起點，注意不能為(0, 0)
	h = 0 #高度補償
	flag = time.time()
	estimated_tag_pos = multiAnchorPositioning(anchor_pos, true_dist, last_pos, h)
	t = time.time() - flag
	print("true_dist:\n", true_dist)
	print("true_dist_diff:\n", true_dist_diff)
	print("estimated_tag_pos:\n", estimated_tag_pos)
	print("spend time:\n",t)