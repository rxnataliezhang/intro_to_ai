import numpy as np
import sys
f_index = 0 

class Factor():
    def __init__(self, variables, var_value, factor_value, index):
        self.variables = variables
        self.var_value = var_value
        self.factor_value = factor_value
        self.index = index
        
class Evidence():
    def __init__(self, variable, value):
        self.variable = variable
        self.value = value
        

def restrict(factor, var, val):
    '''restrict variable var to value val '''
    global f_index
    f_index = f_index + 1
    var_list = factor.variables
    var_loc = var_list.index(var)
    val_loc = factor.var_value[var_loc].index(val)
    k = len(var_list)
    slc = [slice(None)]*k
    slc[var_loc] = val_loc
    factor.variables.remove(var)
    del factor.var_value[var_loc]

    return Factor(factor.variables, factor.var_value, factor.factor_value[slc], f_index)

def multiply(factor1, factor2):
    '''append 1 to variables not involved and do multiplication'''
    ## fix related variables and corresponding values
    var_f1 = factor1.variables
    var_f2 = factor2.variables
    val_f1 = [tuple(l) for l in factor1.var_value]
    val_f2 = [tuple(l) for l in factor2.var_value]
    temp1 = zip(var_f1, val_f1)
    temp2 = zip(var_f2, val_f2)
    temp = list(set(temp1 + temp2))
    temp.sort(key=lambda tup: tup[0])
    new_variables = list(zip(*temp)[0])
    new_var_val = list(zip(*temp)[1])

    ## fix factor value 
    reshape_list1 = []
    reshape_list2 = []
    for x in new_variables:
        if x in var_f1: 
            var_loc1 = var_f1.index(x)       
            reshape_list1.append(len(factor1.var_value[var_loc1]))
        else: reshape_list1.append(1)
    
        if x in var_f2: 
            var_loc2 = var_f2.index(x)
            reshape_list2.append(len(factor2.var_value[var_loc2]))
        else: reshape_list2.append(1)

    fac_val_f1 = factor1.factor_value
    fac_val_f1 = fac_val_f1.reshape(reshape_list1)
    fac_val_f2 = factor2.factor_value
    fac_val_f2 = fac_val_f2.reshape(reshape_list2)
    new_fac_val = fac_val_f1 * fac_val_f2

    return Factor(new_variables, new_var_val, new_fac_val, f_index)

def sumout(factor, var):
    '''summout variable var and return new factor'''
    global f_index
    f_index = f_index + 1
    var_list = factor.variables
    var_loc = var_list.index(var)
    new_fac_val = np.sum(factor.factor_value, axis= var_loc)

    var_list.remove(var)
    del factor.var_value[var_loc]

    return Factor(var_list, factor.var_value, new_fac_val, f_index)

def normalize(factor):
    '''normalize the final factor'''
    global f_index
    f_index = f_index + 1
    fac_value = factor.factor_value
    value_sum = np.sum(fac_value)
    fac_value = np.true_divide(fac_value, value_sum)

    return Factor(factor.variables, factor.var_value, fac_value, f_index)

    
def inference(factorList, queryVariables, orderedListOfHiddenVars, evidenceList):
    '''main variable elimiation algorithm'''
    global f_index
    ###################################################################
    ############ print the process of variable elimination ############
    ###################################################################
    step = 0
    f_index = len(factorList)
    ###################################################################

    for evidence in evidenceList:
        ## restrict factorList with evidenceList
        var = evidence.variable
        ###################################################################
        ############ print the process of variable elimination ############
        ###################################################################
        step = step + 1
        print 'Step', step, 'restrict' , var[1], 'to', evidence.value
        ###################################################################
        fac_invloved = []
        for fac in factorList:
            if var in fac.variables:   
                fac_invloved.append(fac)
            
        for fac in fac_invloved:
            ###################################################################
            ############ print the process of variable elimination ############
            ###################################################################
            sys.stdout.write('f')
            sys.stdout.write(str(fac.index))
            sys.stdout.write('(')
            for i in range(len(fac.variables)-1):
                sys.stdout.write(fac.variables[i][1])
                sys.stdout.write(', ')
            sys.stdout.write(fac.variables[-1][1])
            sys.stdout.write(')')
            sys.stdout.write(' resrticts to ')
            ###################################################################

            new_fac = restrict(fac, var, evidence.value)
            factorList.remove(fac)
            factorList.append(new_fac)

            ###################################################################
            ############ print the process of variable elimination ############
            ###################################################################
            sys.stdout.write('f')
            sys.stdout.write(str(new_fac.index))
            sys.stdout.write('(')
            if len(new_fac.variables) > 0:
                for i in range(len(new_fac.variables)-1):
                    sys.stdout.write(new_fac.variables[i][1])
                    sys.stdout.write(', ')
                sys.stdout.write(new_fac.variables[-1][1])
            print ')'
        print '\n'
        ###################################################################

    for var in orderedListOfHiddenVars:
        ## find the corresponding factor in factorList
        ###################################################################
        ############ print the process of variable elimination ############
        ###################################################################
        step = step + 1
        print 'Step', step, 'sumout variable:', var[1]
        sys.stdout.write('remove factors: ') 
        ###################################################################

        factor_invloved = []
        for i in range(len(factorList)):
            fac = factorList[i]
            if var in fac.variables:
                factor_invloved.append(fac)

                ###################################################################
                ############ print the process of variable elimination ############
                ###################################################################
                sys.stdout.write('f')
                sys.stdout.write(str(fac.index))
                sys.stdout.write('(')
                if len(fac.variables) > 0:
                    for i in range(len(fac.variables)-1):
                        sys.stdout.write(fac.variables[i][1])
                        sys.stdout.write(', ')
                    sys.stdout.write(fac.variables[-1][1])
                sys.stdout.write(') ')
                ###################################################################

                if len(factor_invloved) == 1: 
                    multiply_result = fac 
                else:
                    multiply_result = multiply(multiply_result, fac)
        
        ## sumout var and create new factor
        new_factor = sumout(multiply_result, var)

        ## delete old factors
        for fac in factor_invloved:
            factorList.remove(fac)
        ## add new factor in factorList
        factorList.append(new_factor)

        ###################################################################
        ############ print the process of variable elimination ############
        ###################################################################
        print ''
        sys.stdout.write('add new factor: ')
        sys.stdout.write('f')
        sys.stdout.write(str(new_factor.index))
        sys.stdout.write('(')
        if len(new_factor.variables) > 0:
            for i in range(len(new_factor.variables)-1):
                sys.stdout.write(new_factor.variables[i][1])
                sys.stdout.write(', ')
            sys.stdout.write(new_factor.variables[-1][1])
        print ')'
        print '\n'
        ###################################################################

    ## multiply remaining factors
    ###################################################################
    ############ print the process of variable elimination ############
    ###################################################################
    step = step + 1
    print 'Step', step, 'multiply remaining factors'
    for fac in factorList:
        sys.stdout.write('f')
        sys.stdout.write(str(fac.index))
        sys.stdout.write('(')
        if len(fac.variables) > 0:
            for i in range(len(fac.variables)-1):
                sys.stdout.write(fac.variables[i][1])
                sys.stdout.write(', ')
            sys.stdout.write(fac.variables[-1][1])
        sys.stdout.write(') ')
    ###################################################################
    f_index = f_index + 1
    result = reduce(multiply, factorList)

    ###################################################################
    ############ print the process of variable elimination ############
    ###################################################################
    print ''
    sys.stdout.write('to get: ')
    sys.stdout.write('f')
    sys.stdout.write(str(result.index))
    sys.stdout.write('(') 
    sys.stdout.write(result.variables[0][1])
    sys.stdout.write(')') 
    print '\n'
    ###################################################################

    ## normalize result
    ###################################################################
    ############ print the process of variable elimination ############
    ###################################################################
    step = step + 1
    print 'Step', step, 'normalize'
    print 'Before normalizion'
    print result.variables[0][1], result.factor_value
    ###################################################################

    network_output = normalize(result)

    return network_output
              

def variable_elimination(queryVariables, evidenceList):
    ## factorList
    allvariables =[(0,'Trav'), (1,'FP'),  (2, 'Fraud'), (3,'IP'), (4, 'OC'), (5, 'CRP')]

    f1 = Factor([(0,'Trav')],\
                [[True, False]], \
                np.array([0.05, 0.95]), 1)
    f2 = Factor([(0,'Trav'), (2, 'Fraud')], \
                [[True, False], [True, False]], \
                np.array([[0.01, 0.99], [0.004, 0.996]]), 2)
    f3 = Factor([(0,'Trav'), (1,'FP'), (2, 'Fraud')], \
                [[True, False], [True, False], [True, False]], \
                np.array([[[0.9, 0.9], [0.1, 0.1]],[[0.1, 0.01], [0.9, 0.99]]]), 3)
    f4 = Factor([(4, 'OC')], \
                [[True, False]], \
                np.array([0.8, 0.2]), 4)
    f5 = Factor([(2, 'Fraud'), (3,'IP'), (4, 'OC')], \
                [[True, False], [True, False], [True, False]], \
                np.array([[[0.15, 0.051], [0.85, 0.949]], [[0.1, 0.001], [0.9, 0.999]]]), 5)
    f6 = Factor([(4, 'OC'), (5, 'CRP')], \
                [[True, False], [True, False]], \
                np.array([[0.1, 0.9], [0.01, 0.99]]), 6)

    factorList = [f1, f2, f3, f4, f5, f6]
    
    ## orderedListOfHiddenVars
    orderedListOfHiddenVars = allvariables

    evidenceVariables = []
    for e in evidenceList:
        evidenceVariables.append(e.variable)

    orderedListOfHiddenVars.remove(queryVariables)

    for var in evidenceVariables:
        orderedListOfHiddenVars.remove(var)

    output = inference(factorList, queryVariables, orderedListOfHiddenVars, evidenceList)
    
    return output

def test():
    ## input evidence
    e1 = Evidence((1, 'FP'), True)
    e2 = Evidence((3, 'IP'), False)
    e3 = Evidence((5, 'CRP'), True)

    evidenceList = [e1, e2, e3]
    queryVariables = (2, 'Fraud')

    result = variable_elimination(queryVariables, evidenceList)
    print 'After normaliztion, the final result:'
    print result.variables[0][1], result.factor_value

if __name__ == '__main__':
    test()
    
    





        
    

    


 
