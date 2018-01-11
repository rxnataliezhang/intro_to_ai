import time
import random
import numpy as np
from collections import OrderedDict
counter = 0

class Variable():
    def __init__(self, location, value, domain):
        self.location = location
        self.value = value
        self.curdomain = domain
        self.reomoved = []

    def find_value(self):
        '''return value'''
        return(self.value)

    def find_location(self):
        '''return location'''
        return(self.location)

    def find_row(self):
        '''return row'''
        return(self.location[0])   

    def find_column(self):
        '''return column'''
        return(self.location[1])        

    def find_curr_domain(self):
        '''return current domain'''
        return(self.curdomain)


def backtracking_search(variables, matrix, fc, mcv, lcv):
    assignment = {}

    for var in variables:
        value = var.find_value()
        if value != 0:
            location = var.find_location()
            assignment[location] = value
            remove_domain(value, var, assignment, variables, matrix, fc)

    return recursive_backtracking(assignment, variables, matrix, fc, mcv, lcv)

def recursive_backtracking(assignment, variables, matrix, fc, mcv, lcv):
    global counter

    if len(assignment) == len(variables):
        return assignment

    if ifempty(variables, fc) == True:
        return None

    curr_var = select_unassigned_variable(assignment, variables, mcv)
    for val in order_domain_values(curr_var, assignment, variables, matrix, lcv):
        counter += 1
        if fc == True or conflict(val, curr_var, assignment) == False:
            add_assignment(val, curr_var, assignment, variables)
            revised_variables = remove_domain(val, curr_var, assignment, variables, matrix, fc)
            result = recursive_backtracking(assignment, variables, matrix, fc, mcv, lcv)
            if result is not None:
                return result
            remove_assignment(val, curr_var, assignment, variables)
            add_domain(val, revised_variables, fc)

    return None

def select_unassigned_variable(assignment, variables, mcv):
    '''Choose next unassigned variable. If mcv is False, choose variable in order of variables'''
    if mcv == False: 
        for var in variables:
            if var.find_location() not in assignment.keys():
                return var

    else:
        variables_mrv = most_constrained_variable(assignment, variables)
        if len(variables_mrv) == 1:
            return variables_mrv[0]
        else:
            variables_mcv = most_constraining_variable(assignment, variables_mrv)
            if len(variables_mcv) == 1:
                return variables_mcv[0]
            else:
                '''If exisiting tie after two heuristics, choose variable randomly'''
                next_variable = random.choice(variables_mcv)
                return next_variable
                
def most_constrained_variable(assignment, variables):
    '''Apply most constrained variable and return variables with minimum number of domain'''
    variables_mrv = []
    min_num_domain = 9

    for var in variables:
        if var.find_location() not in assignment.keys():
            num_domain = len(var.find_curr_domain())
            if min_num_domain > num_domain:
                min_num_domain = num_domain
                variables_mrv = []
                variables_mrv.append(var)
            elif min_num_domain == num_domain:
                variables_mrv.append(var)

    return variables_mrv
        
def most_constraining_variable(assignment, variables):
    '''Apply most constraining variable and return variables with most potential constraints'''
    variables_mcv = []
    max_num_constr = 0

    for var in variables:
        num_constr = 0
        neighbours = find_neighbours(var)
        for each_neigh in neighbours:
            if each_neigh not in assignment.keys():
                num_constr += 1 
        if max_num_constr < num_constr:
            max_num_constr = num_constr
            variables_mcv = []
            variables_mcv.append(var)
        elif max_num_constr == num_constr:
            variables_mcv.append(var)

    return variables_mcv

def order_domain_values(var, assignment, variables, matrix, lcv):
    '''Sort the domain of current variable. If lcv is False, keep original order'''
    if lcv == False:
        return var.find_curr_domain()

    else:
        '''sort values with least constraining value'''
        neighbours = find_neighbours(var)
        value_list = var.find_curr_domain()
        dic_value_num_constr= {}

        for each_value in value_list:
            num_value_constr = 0
            for each_neigh in neighbours:
                each_neigh_domain = matrix[each_neigh[0]][each_neigh[1]].find_curr_domain()
                if each_value in each_neigh_domain:
                    num_value_constr += 1
            dic_value_num_constr[each_value] = num_value_constr
        sorted(dic_value_num_constr.values())
        value_lcv = dic_value_num_constr.keys()

        return value_lcv
       
def row_conflict(val, var, assignment):
    '''Check row conflict'''
    row = var.find_row()
    col = var.find_column()

    for j in range(9):
        if (row,j) in assignment.keys() and val == assignment[(row,j)]: 
            return True

    return False
            
def col_conflict(val, var, assignment):
    '''Check column conflict'''
    row = var.find_row()
    col = var.find_column()

    for i in range(9):
        if (i, col) in assignment.keys() and val == assignment[(i, col)]:
            return True
            
    return False

def block_conflict(val, var, assignment):
    '''Check block conflict'''
    row = var.find_row()
    col = var.find_column()
    row_block = row/3 * 3
    col_block = col/3 * 3

    for i in range(row_block, row_block+3):
        for j in range(col_block, col_block+3):
            if (i, j) in assignment.keys() and val == assignment[(i,j)]:
                return True

    return False

def conflict(val, var, assignment):
    '''If any conflict in row, column or block, return True'''
    rowcon = row_conflict(val, var, assignment)
    colcon = col_conflict(val, var, assignment)
    blockcon = block_conflict(val, var, assignment)

    return rowcon or colcon or blockcon
    
def add_assignment(val, var, assignment, variables):
    '''Add an assignment to the assignment dict'''
    location = var.find_location()
    assignment[location] = val


def remove_assignment(val, var, assignment, variables):
    '''Prune an assignment from the assignment dict'''
    location = var.find_location()
    del assignment[location]

def remove_domain(val, var, assignment, variables, matrix, fc):
    '''Remove the equal values from the current domain of all variables related'''
    if fc == True:
        counter_domain = 0
        revised_variables = []
        neighbours = find_neighbours(var)
        for loc in neighbours:
            each_var = matrix[loc[0]][loc[1]]
            if loc not in assignment.keys() and val in each_var.find_curr_domain():
                revised_variables.append(each_var)
                each_var.curdomain.remove(val)
        return revised_variables

def add_domain(val, revised_variables, fc):
    '''Add the original values to the current domain of all variables related'''
    if fc == True:
        for each_var in revised_variables:
            each_var.curdomain.append(val)

def find_neighbours(var):
    '''Find neighbours of variable , i.e. in same row, same column or same block'''
    neighbours = set()
    row = var.find_row()
    col = var.find_column()

    for i in range(9):
        neighbours.add((row,i))
        neighbours.add((i,col))
    row_block = row/3 * 3
    col_block = col/3 * 3
    for i in range(row_block, row_block+3):
        for j in range(col_block, col_block+3):
            neighbours.add((i,j))
    neighbours.remove((row,col))
    return neighbours

def ifempty(variables, fc):
    '''Check if the domains of reamining variables empty if empty return none'''
    if fc == True:
        for var in variables:
            if var.find_curr_domain() == []:  
                return True
        return False
        

def initialize_game(board, fc, mcv, lcv):
    '''Assign given values and create variables list'''
    variables = []
    matrix = []
    # counter = 0

    for row in range(9):
        temp_matrix = []
        for col in range(9):
            if board[row][col] == 0:
                var = Variable((row, col), board[row][col], list(range(1,10)))
            else:
                var = Variable((row, col), board[row][col], [board[row][col]])
            variables.append(var)
            temp_matrix.append(var)
        matrix.append(temp_matrix)
    
    final_board = backtracking_search(variables, matrix, fc, mcv, lcv)
    return (final_board, counter)


easy = \
    [[0,6,1,0,0,0,0,5,2],
     [8,0,0,0,0,0,0,0,1],
     [7,0,0,5,0,0,4,0,0],
     [9,0,3,6,0,2,0,4,7],
     [0,0,6,7,0,1,5,0,0],
     [5,7,0,9,0,3,2,0,6],
     [0,0,4,0,0,9,0,0,5],
     [1,0,0,0,0,0,0,0,8],
     [6,2,0,0,0,0,9,3,0]]

medium = \
   [[5,0,0,6,1,0,0,0,0],
    [0,2,0,4,5,7,8,0,0],
    [1,0,0,0,0,0,5,0,3],
    [0,0,0,0,2,1,0,0,0],
    [4,0,0,0,0,0,0,0,6],
    [0,0,0,3,6,0,0,0,0],
    [9,0,3,0,0,0,0,0,2],
    [0,0,6,7,3,9,0,8,0],
    [0,0,0,0,8,6,0,0,5]]

hard = \
   [[0,4,0,0,2,5,9,0,0],
    [0,0,0,0,3,9,0,4,0],
    [0,0,0,0,0,0,0,6,1],
    [0,1,7,0,0,0,0,0,0],
    [6,0,0,7,5,4,0,0,9],
    [0,0,0,0,0,0,7,3,0],
    [4,2,0,0,0,0,0,0,0],
    [0,9,0,5,4,0,0,0,0],
    [0,0,8,9,6,0,0,5,0]]

evil = \
   [[0,6,0,8,2,0,0,0,0],
    [0,0,2,0,0,0,8,0,1],
    [0,0,0,7,0,0,0,5,0],
    [4,0,0,5,0,0,0,0,6],
    [0,9,0,6,0,7,0,3,0],
    [2,0,0,0,0,1,0,0,7],
    [0,2,0,0,0,9,0,0,0],
    [8,0,4,0,0,0,7,0,0],
    [0,0,0,0,4,8,0,2,0]]


################################################################
######### Print solution to each test puzzle
def test():
    start_time = time.time()
    final_result = initialize_game(medium, fc=True, mcv=True, lcv=True)
    end_time = time.time()
    print("--- %s seconds ---" % (end_time - start_time))

    print "Iterative number is " + str(final_result[1])
    result_matrix = [[0 for x in range(9)] for y in range(9)] 
    for point in final_result[0].keys():
        result_matrix[point[0]][point[1]] = final_result[0][point]
    print "----------------------------"
    for x in range(9):
        print result_matrix[x]
    print "----------------------------"


################################################################
######### Measure running time and # of nodes expanded
def measure():
    global counter
    board_list = [("easy", easy), 
                ("medium", medium), 
                ("hard", hard), 
                ("evil", evil)]

    running_time_B = OrderedDict()
    running_time_B_F = OrderedDict()
    running_time_B_F_H = OrderedDict()
    node_counter_B = OrderedDict()
    node_counter_B_F = OrderedDict()
    node_counter_B_F_H = OrderedDict()

    for board in board_list:
        running_time_B[board[0]]=[]
        node_counter_B[board[0]]=[]
        running_time_B_F[board[0]]=[]
        node_counter_B_F[board[0]]=[]
        running_time_B_F_H[board[0]]=[]
        node_counter_B_F_H[board[0]]=[]

        for x in range(50):
            start_time = time.time()
            counter = 0
            result_B = initialize_game(board[1], fc=False, mcv=False, lcv=False)
            end_time = time.time()
            running_time_B[board[0]].append(end_time - start_time)
            node_counter_B[board[0]].append(result_B[1])

            start_time = time.time()
            counter = 0
            result_B_F = initialize_game(board[1], fc=True, mcv=False, lcv=False)
            end_time = time.time()
            running_time_B_F[board[0]].append(end_time - start_time)
            node_counter_B_F[board[0]].append(result_B_F[1])

            start_time = time.time()
            counter = 0
            result_B_F_H = initialize_game(board[1], fc=True, mcv=True, lcv=True)
            end_time = time.time()
            running_time_B_F_H[board[0]].append(end_time - start_time)
            node_counter_B_F_H[board[0]].append(result_B_F_H[1])

    print "running time: "
    for x in running_time_B.keys():
        np_run_time = np.array(running_time_B[x])
        run_time_avr = np.mean(np_run_time)
        run_time_std = np.std(np_run_time)
        print 'B     | average |' + ' ' + x + ' | ' + str(run_time_avr) 
        print 'B     |   std   |' + ' ' + x + ' | ' + str(run_time_std) 

    for x in running_time_B_F.keys():
        np_run_time = np.array(running_time_B_F[x])
        run_time_avr = np.mean(np_run_time)
        run_time_std = np.std(np_run_time)
        print 'B_F   | average |' + ' ' + x + ' | ' + str(run_time_avr)
        print 'B_F   |   std   |' + ' ' + x + ' | ' + str(run_time_std) 

    for x in running_time_B_F_H.keys():
        np_run_time = np.array(running_time_B_F_H[x])
        run_time_avr = np.mean(np_run_time)
        run_time_std = np.std(np_run_time)
        print 'B_F_H | average |' + ' ' + x + ' | ' + str(run_time_avr)
        print 'B_F_H |   std   |' + ' ' + x + ' | ' + str(run_time_std) 

    print"\n"
    print "num of nodes: "
    for x in node_counter_B.keys():
        np_node = np.array(node_counter_B[x])
        np_node_avr = np.mean(np_node)
        np_node_std = np.std(np_node)
        print 'B     | average |' + ' ' + x + ' | ' + str(np_node_avr) 
        print 'B     |   std   |' + ' ' + x + ' | ' + str(np_node_std)

    for x in node_counter_B_F.keys():
        np_node = np.array(node_counter_B_F[x])
        np_node_avr = np.mean(np_node)
        np_node_std = np.std(np_node)
        print 'B_F   | average |' + ' ' + x + ' | ' + str(np_node_avr) 
        print 'B_F   |   std   |' + ' ' + x + ' | ' + str(np_node_std) 

    for x in node_counter_B_F_H.keys():
        np_node = np.array(node_counter_B_F_H[x])
        np_node_avr = np.mean(np_node)
        np_node_std = np.std(np_node)
        print 'B_F_H | average |' + ' ' + x + ' | ' + str(np_node_avr) 
        print 'B_F_H |   std   |' + ' ' + x + ' | ' + str(np_node_std) 

if __name__ == '__main__':
    # test()
    measure()
