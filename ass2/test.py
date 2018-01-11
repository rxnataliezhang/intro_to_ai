from ass2 import *

def case_b2():
    ## input evidence
    e1 = Evidence((1, 'FP'), True)
    e2 = Evidence((3, 'IP'), False)
    e3 = Evidence((5, 'CRP'), True)

    evidenceList = [e1, e2, e3]
    queryVariables = (2, 'Fraud')

    result = variable_elimination(queryVariables, evidenceList)
    print 'After normaliztion, the final result:'
    print result.variables[0][1], result.factor_value

def case_c():
    ## input evidence
    e1 = Evidence((1, 'FP'), True)
    e2 = Evidence((3, 'IP'), False)
    e3 = Evidence((5, 'CRP'), True)
    e4 = Evidence((0, 'Trav'), True)

    evidenceList = [e1, e2, e3, e4]
    queryVariables = (2, 'Fraud')

    result = variable_elimination(queryVariables, evidenceList)
    print 'After normaliztion, the final result:'
    print result.variables[0][1], result.factor_value

def case_b1():
    queryVariables = (2, 'Fraud')
    evidenceList = []
    result = variable_elimination(queryVariables, evidenceList)
    print 'After normaliztion, the final result:'
    print result.variables[0][1], result.factor_value

def case_d1():
    queryVariables = (2, 'Fraud')
    e1 = Evidence((3, 'IP'), True)
    evidenceList = [e1]
    result = variable_elimination(queryVariables, evidenceList)
    print 'After normaliztion, the final result:'
    print result.variables[0][1], result.factor_value

def case_d2():
    queryVariables = (2, 'Fraud')
    e1 = Evidence((3, 'IP'), True)
    e2 = Evidence((5, 'CRP'), True)
    evidenceList = [e1, e2]
    result = variable_elimination(queryVariables, evidenceList)
    print 'After normaliztion, the final result:'
    print result.variables[0][1], result.factor_value

def case_d3():
    queryVariables = (2, 'Fraud')
    e1 = Evidence((3, 'IP'), True)
    e2 = Evidence((5, 'CRP'), False)
    evidenceList = [e1, e2]
    result = variable_elimination(queryVariables, evidenceList)
    print 'After normaliztion, the final result:'
    print result.variables[0][1], result.factor_value 

if __name__ == '__main__':
    print 'b1: P(Fraud)'
    case_b1()
    print '----------------------------'
    print 'b2: P(Fruad|fp,~ip,crp)'
    case_b2()
    print '----------------------------'
    print 'c:  P(Fruad|fp,~ip,crp,trav)'
    case_c()
    print '----------------------------'
    print 'd1: P(Fruad|ip)'
    case_d1()
    print '----------------------------'
    print 'd2: P(Fruad|ip,crp)'
    case_d2()