import numpy as np
import copy
import random
from gridWorld import *
import matplotlib.pyplot as plt
import copy
gamma_value = 0.99
epsilon_value = 0.05*100
random.seed(1)
np.random.seed(1)

def q_learning(tran_matrix, r_array):

    episode = 0
    # q_matrix = np.zeros((16,4))
    q_matrix = np.random.uniform(-1, 1, size = (16,4))
    q_matrix[15] = 100
    n_matrix = np.zeros((16,4))
    a_matrix = np.zeros((16,1)) + 7
    while episode < 10000:
        last_a_matrix = copy.deepcopy(a_matrix)
        state = 4
        flag = True
        episode += 1
        while flag:
            q_matrix, state, a_matrix = q_learning_loop(q_matrix, state, tran_matrix, r_array, n_matrix, a_matrix)
            if state == 15: 
                flag = False
        change = np.any(np.negative(a_matrix == last_a_matrix))
        if change == True: convergence = episode
    print 'convergence episode', convergence
    return q_matrix, a_matrix

def q_learning_loop(q_matrix, state, tran_matrix, r_array, n_matrix, a_matrix):

    ## choose next action
    rand1 = random.randint(1,100)
    if rand1 <= epsilon_value: next_action = random.randint(0,3)
    else: next_action = np.argmax(q_matrix[state])
    n_matrix[state, next_action] += 1

    ## execute action and get the next state
    next_state_list = tran_matrix[state,:,next_action] * 100
    rand2 = random.randint(1,100)
    summation = 0
    next_state = 0
    for i in range(next_state_list.shape[0]):
        summation = summation + next_state_list[i]
        if summation > rand2: 
            next_state = i
            break

    ## receive immediate reward
    reward = r_array[state]

    ## update q_matrix and a_matrix
    alpha = 1 / float(n_matrix[state, next_action])
    a_matrix[state] = np.argmax(q_matrix[state])
    q_matrix[state, next_action] = q_matrix[state, next_action] + alpha * \
            (reward + gamma_value*q_matrix[next_state].max() - q_matrix[state, next_action]) 
    a_matrix[state] = np.argmax(q_matrix[state])
    

    ## update state
    state = next_state
    return q_matrix, state, a_matrix


tran_matrix, r_array = gridWorld()
q_matrix, a_matrix = q_learning(tran_matrix, r_array)

optimal_action = []
for x in range(len(a_matrix)):
    if a_matrix[x]==0: optimal_action.append('u')
    elif a_matrix[x]==1: optimal_action.append('d')
    elif a_matrix[x]==2: optimal_action.append('l')
    elif a_matrix[x]==3: optimal_action.append('r')
    else: optimal_action.append('g')
optimal_action = np.array(optimal_action)
optimal_action.resize(4,4)

print 'epsilon =', epsilon_value / 100
print 'Q matrix'
print q_matrix
print 'optimal action'
print optimal_action



    