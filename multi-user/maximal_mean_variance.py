import sys
import math
import numpy as np
from numpy.linalg import inv

# This is a C-like python...
#

current_mean=[[0 for u in range(100)] for v in range(5)]
cost = [[0 for u in range(100)] for v in range(5)]
pre_mean=[[0 for u in range(100)] for v in range(5)] 
prior_mean = [[0 for u in range(100)] for v in range(5)]
kernel = [[[ 0 for u in range(100) ] for v in range(100)] for w in range(5)]
current_var = [[0 for u in range(200)] for v in range(5)]
pre_var = [[0 for u in range(200)] for v in range(5)]
observe_accu = [[ 0 for u in range(100) ] for v in range(5)]
has_chosen = [[0 for u in range(100)] for v in range(5)]
has_run = [[0 for u in range(500)] for v in range(5)]
has_sample = [0 for u in range(5)]
last_run_time = [0 for u in range(5)]
run_times =[0 for u in range(5)]
run_time = [[0 for u in range(500)] for v in range(5)] #run_time[i][j]: user i's j-th running is at time t, j >=1
last_run_algo=[-1 for u in range(5)]
second_last_run_algo=[-1 for v in range(5)]
accuracy=[0 for u in range(5)]
sum_accuracy=0

a = [[-1 for u in range(2000)] for v in range(5)]
y = [-1 for u in range(2000)]
beta = [0 for u in range(2000)]
sigma_t_k = [[0 for u in range(2000)] for v in range(5)]
sigma_t = [[[0 for u in range(2000)] for v in range(2000)] for v in range(5)]
c_t = [[0 for u in range(2000)] for v in range(2000)]

max_cost = 0
pi = 3.14159265359

def init():

	global current_mean, current_var, chosen, a, y, beta, sigma_t_k, sigma_t, c_t
	for j in range(5):
		last_run_time[j] = 0
		run_times[j] = 0
		
		for i in range(100):
			current_mean[j][i] = prior_mean[j][i]
			pre_mean[j][i] = prior_mean[j][i]
			current_var[j][i] = kernel[j][i][i]
			pre_var[j][i] = kernel[j][i][i]
			has_run[j][i] = 0
	return

f_kernel = open("kernel.txt",'r')
f_mean = open("mean.txt",'r')
f_test = open("test.txt",'r')
f_cost = open("cost.txt",'r')
fout = open("maximal_mean_variance.txt",'w')
fout2=open("sum_accuracy_algo5.txt",'w')
fout3 = open("debug_maximal_mean_variance.txt",'w')
fout4 = open("post_mean.txt",'w')


j = 0
v = 0
for line in f_kernel:
	line = line.strip(" \n").split(" ")
	for i in range(len(line)):
		kernel[v][j][i] = float(line[i])
	j = j+1
	if j == 100:
		v = v + 1
		j = 0

j = 0
for line in f_mean:
	line = line.strip(" \n").split(" ")
	for i in range(len(line)):
		prior_mean[j][i] = float(line[i])
	j = j + 1

j = 0
for line in f_test:
	line = line.strip(" \n").split(" ")
	for i in range(len(line)):
		observe_accu[j][i] = float(line[i])
	j = j + 1

j = 0
for line in f_cost:
	line = line.strip(" \n").split(" ")
	for i in range(len(line)):
		cost[j][i] = float(line[i])
		if cost[j][i] > max_cost:
			max_cost = cost[j][i]
	j = j + 1
 #-----------------------------------------


t = 1
max_iteration = 500
init()

while t <= 500:
	for i in range(5):  # for every user, we choose the best algorithm

		max_ = -100000.0
		algo_run = -5
		beta[t] = 2 * max_cost * math.log(pi * pi * 100 * t * t * 1.0 / (0.5 * 6));
		beta[0] = beta[1]
#-----------------------------------------use an algo --------------------------------
		tmp_reward=0

		for algo in range(100):
			aa = math.sqrt(beta[last_run_time[i]] * 1.0 / cost[i][algo])
			
			b = math.sqrt(current_var[i][algo])
			tmp3 =  aa * b
			tmp_reward = current_mean[i][algo] + tmp3
			if tmp_reward > max_: # 
				max_ = tmp_reward
				algo_run = algo
		has_chosen[i][algo_run] = has_chosen[i][algo_run]+1
		a[i][t] = algo_run # For user i, at time stamp t, choose algorithm a[i][t], t >= 1
		if i == 0:
			fout3.write("T = " + str(t) + ",algorithm "+str(algo_run)+" , tmp_reward is "+str(max_)+"\n")
	
	user_id = -1
	max_mean_variance = -10000000.0

	# Then pick up the user with the maximal relative information gain between the two time stamps it runs
	for i in range(5):
		if t > 5: # have only run one algorithm
			a1 = current_mean[i][last_run_algo[i]] + math.sqrt(beta[last_run_time[i]] * 1.0 / cost[i][last_run_algo[i]]) * math.sqrt(current_var[i][last_run_algo[i]])
			a2 = pre_mean[i][second_last_run_algo[i]] + math.sqrt(beta[last_run_time[i]-1] * 1.0 / cost[i][second_last_run_algo[i]]) * math.sqrt(pre_var[i][second_last_run_algo[i]])
			#mean_variance = math.sqrt(beta[last_run_time[i]] * 1.0 / cost[i][last_run_algo[i]]) * math.sqrt(current_var[i][last_run_algo[i]])
			
			mean_variance = a2-a1
									
			#a1 < a2 if a1 - a2

			#a1 > a2 if a2 - a1
			fout3.write("user "+str(i)+", current= "+str(a1)+" ,prev= "+str(a2)+"\n")
			fout3.write("current mean = "+str(current_mean[i][last_run_algo[i]])+", current var is: " + str(current_var[i][last_run_algo[i]])+"\n")
			fout3.write("prev mean = "+str(pre_mean[i][second_last_run_algo[i]])+", prev var is: " + str(pre_var[i][second_last_run_algo[i]])+"\n")
			if mean_variance > max_mean_variance:
				max_mean_variance = mean_variance
				user_id = i
			fout3.write("pick up user "+str(user_id)+"\n")
		

	if t == 1:
		user_id = 0
	elif t == 2:
		user_id = 1
	elif t == 3:
		user_id = 2
	elif t == 4:
		user_id = 3
	elif t == 5:
		user_id = 4
	elif t == 6:
		user_id = 0
	elif t == 7:
		user_id = 1
	elif t == 8:
		user_id = 2
	elif t == 9:
		user_id = 3
	elif t == 10:
		user_id = 4
	
	
	fout3.write(str(user_id)+"\n")
	has_sample[user_id] += 1;
	has_run[user_id][a[user_id][t]] += 1;
	run_times[user_id] += 1;
	print (str(user_id)+ " has run " + str(run_times[user_id]) + " times.")
	run_time[user_id][run_times[user_id]] = t;
	if (run_times[user_id] >=2):
		second_last_run_algo[user_id] = last_run_algo[user_id];
	last_run_algo[user_id] = a[user_id][t];
	last_run_time[user_id] = t;
	y[t] = observe_accu[user_id][a[user_id][t]] + 0.01
	accuracy[user_id] = observe_accu[user_id][a[user_id][t]]

	if a[user_id][t] != -5:
		fout.write(str(user_id) + " " + str(a[user_id][t]) + "\n");
		for v in range(5):
			sum_accuracy = sum_accuracy + accuracy[v]	
		#fout2.write(str(user_id)+" "+str(a[user_id][t])+" "+str(sum_accuracy)+"\n")
			fout2.write(str(cost[v][last_run_algo[v]])+ " " + str(accuracy[v])+" ")
		fout2.write("\n")
	

#--------------------------------------------update----------------------------------

	for u in range(5):
		if u == user_id:
			for i in range(run_times[u]):
				for j in range(run_times[u]):
					sigma_t[u][i][j] = kernel[u][a[u][run_time[u][i+1]]][a[u][run_time[u][j+1]]]
					if i == j:
						sigma_t[u][i][j] = sigma_t[u][i][j] + 0.01 * 0.01
			x=[]
			for v in range(run_times[u]):
				x.append([sigma_t[u][v][:run_times[u]]])
			c_t = inv(np.asarray(x).reshape(run_times[u],run_times[u]))

			y_tmp=[]
			m_tmp=[]
			tmp3=[]
			for i in range(run_times[u]):
				y_tmp.append(y[run_time[u][i+1]])
				m_tmp.append(prior_mean[u][a[u][run_time[u][i+1]]])
				tmp3.append(y[run_time[u][i+1]] - prior_mean[u][a[u][run_time[u][i+1]]])

			for i in range(100):
				for j in range(run_times[u]):

					# For every user, every algorithm, we have such a vector
					sigma_t_k[u][j] = kernel[u][a[u][run_time[u][j+1]]][i]


			
				tmp1 = np.dot(sigma_t_k[u][:run_times[u]],c_t)			
				#tmp2 = np.dot(tmp1, y_tmp)
				tmp2 = np.dot(tmp1, y_tmp)

				pre_mean[u][i] = current_mean[u][i]
				current_mean[u][i] = tmp2 
				# Have confirmed that the update of mean is correct

				update = np.dot(tmp1,sigma_t_k[u][:run_times[u]])
				pre_var[u][i] = current_var[u][i]
				#print ("algorithm " + str(i) + " previous var is: " + str(pre_var[u][i]) + "\n")
				current_var[u][i] = kernel[u][i][i] - update

	t = t + 1
	if t == 500:
		for u in range(5):
			for uu in range(100):
				fout4.write(str(current_mean[u][uu])+" ")
			fout4.write("\n")
	print ("t="+str(t))
	fout.flush()
	stop=1
	for i in range(5):
		for j in range(100):
			if has_run[i][j] == 0:
				stop=0
				break

	# All users have run all algorithms
	if stop == 1:
		break








#