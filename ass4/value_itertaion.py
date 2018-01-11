import numpy as np
import copy
from gridWorld import *
gamma_value = 0.99

def value_iteration(tran_matrix, r_array):
    r_array_temp = np.delete(r_array, -1)
    r_array_temp.resize(4,4)   
    value_matrix = copy.deepcopy(r_array_temp)
    a_matrix = np.zeros((16,1)) + 7
    flag = True
    
    while flag:
        value_matrix_origin = copy.deepcopy(value_matrix)
        value_matrix, a_matrix = value_iteration_loop(value_matrix, tran_matrix, r_array, a_matrix)
        delta_matrix = value_matrix - value_matrix_origin
        if np.any(delta_matrix >= 0.01): continue
        else: flag = False
    return value_matrix, a_matrix

def value_iteration_loop(value_matrix, tran_matrix, r_array, a_matrix):
    new_value_matrix = copy.deepcopy(value_matrix)
    for i in range(len(r_array)-1):
        if i == 15: continue
        m_s = i / 4   ## row of element to update in value_matrix
        n_s = i % 4   ## col of element to update in value_matrix
        value_list = []
        ## find all states related
        state_related = []
        for x in [i-4, i-1, i, i+1, i+4]:
            if x>=0 and x<=15: state_related.append(x)

        for j in range(tran_matrix.shape[2]): ## iterate all the actions
            ## caculate bellman equation
            value = 0
            for x in state_related:
                m_s_p = x / 4
                n_s_p = x % 4
                value = value + gamma_value * tran_matrix[i][x][j] * value_matrix[m_s_p][n_s_p]
            value_list.append(value)
        idx = value_list.index(max(value_list))
        a_matrix[i][0] = idx
        new_value_matrix[m_s][n_s] = max(value_list) + r_array[i]

    return new_value_matrix, a_matrix

tran_matrix, r_array = gridWorld()
value_matrix, a_matrix = value_iteration(tran_matrix, r_array)

optimal_action = []
for x in range(len(a_matrix)):
    if a_matrix[x]==0: optimal_action.append('u')
    elif a_matrix[x]==1: optimal_action.append('d')
    elif a_matrix[x]==2: optimal_action.append('l')
    elif a_matrix[x]==3: optimal_action.append('r')
    else: optimal_action.append('g')
optimal_action = np.array(optimal_action)
optimal_action.resize(4,4)
print 'value matrix'
print value_matrix
print 'optimal action'
print optimal_action



