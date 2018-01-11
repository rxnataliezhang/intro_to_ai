import numpy as np
import matplotlib.pyplot as plt
import math
import time
import copy

class Node():
    def __init__(self, attr, subnode):
        self.attr = attr
        self.subnode = subnode
    def __str__(self):
        return '{0}: {1}  {2}'.format(self.attr.attr_name, \
                (self.subnode[0].attr.attr_name if isinstance(self.subnode[0], Node) else ('Leaf', self.subnode[0].classification)), \
                (self.subnode[1].attr.attr_name if isinstance(self.subnode[1], Node) else ('Leaf', self.subnode[1].classification)))

class Leaf():
    def __init__(self, classification):
        self.classification = classification
    def __str__(self):
        return 'Leaf'+str(self.classification)

class Attribute():
    def __init__(self, attr_index, attr_name):
        self.attr_index = attr_index
        self.attr_name = attr_name
    
def get_attr_list(word_file):
    attribute_list = []
    i = 0
    for line in open(word_file):
        attribute_list.append(Attribute(i, line.strip('\n')))
        i += 1  
    return attribute_list

def get_label_file(label_file):
    lines = [int(line.rstrip('\n')) for line in open(label_file)]
    label = np.array(lines)
    return label

def get_data_file(data_file, n, m):
    data = np.zeros((n,m))
    with open(data_file, 'r') as in_file:
        for line in in_file:
            nums = line.split()
            data[int(nums[0])-1][int(nums[1])-1] = 1
    return data


attribute_list = get_attr_list('words.txt')
attribute_list = np.array(attribute_list)
train_label = get_label_file('trainLabel.txt')
train_data = get_data_file('trainData.txt', len(train_label), len(attribute_list))

test_label = get_label_file('testLabel.txt')
test_data = get_data_file('testData.txt', len(test_label), len(attribute_list))    
    
def decision_tree_learning(data, label, attribute_list, max_depth, depth=0, default=None):
    if depth >= max_depth: return Leaf(get_mode(label))
    depth += 1
    if data.shape[0] == 0:
        return Leaf(default)
    elif same_label(label):
        return Leaf(label[0])
    elif data.shape[1] == 0:
        return Leaf(get_mode(label))
    else:
        best_attr_index = choose_attribute(data, label)
        best_attr = attribute_list[best_attr_index]
        print 'best attribute is', best_attr.attr_name
        mode = get_mode(label)
        subnode = []
        ## build subtree for every value of the attribute
        for value in [0,1]:
            index = split_data(data, value, best_attr_index)
            data_temp = copy.deepcopy(data)
            label_temp = copy.deepcopy(label)
            attribute_list_temp = copy.deepcopy(attribute_list)
            sub_data = data_temp[index]
            sub_label = label_temp[index]
            sub_data = np.delete(sub_data, best_attr_index, axis=1)
            sub_attribute_list = np.delete(attribute_list_temp, best_attr_index)
            subnode.append(decision_tree_learning(sub_data, sub_label, sub_attribute_list, max_depth, depth, mode))
        return Node(best_attr, subnode)

def choose_attribute(data, label):
    information_gain = []
    for eachAttr in data.T:
        c1_f, c2_f, c1_t, c2_t = 0, 0, 0, 0
        l = len(label)
        for i in range(l):
            if label[i] == 1 and eachAttr[i] == 0: c1_f += 1 
            elif label[i] == 2 and eachAttr[i] == 0: c2_f += 1
            elif label[i] == 1 and eachAttr[i] == 1: c1_t += 1 
            else: c2_t += 1
        ig = calentropy(float(c1_t+c2_t)/l, float(c1_f+c2_f)/l) - \
             float(c1_t+c1_f) / l * calentropy(c1_t/float(c1_t+c1_f), c1_f/float(c1_t+c1_f)) - \
             float(c2_t+c2_f) / l * calentropy(c2_t/float(c2_t+c2_f), c2_f/float(c2_t+c2_f))
        information_gain.append(ig)
    print 'information gain is', max(information_gain)
    return information_gain.index(max(information_gain))
    
def calentropy(a, b):
    if a == 0 or b == 0: return 0
    entropy = (-1)* a * math.log(a, 2) + (-1)* b * math.log(b, 2)
    return entropy

def same_label(label):
    '''justify whether the label of all examples same'''
    for i in range(len(label)):
        if label[0] != label[i]:
            return False
    return True

def get_mode(label):
    '''return the majority label of all examples'''
    c1, c2 = 0, 0
    for i in label:
        if i == 1: c1 += 1
        else: c2 += 1
    if c1 >= c2: return 1
    else: return 2 

def split_data(data, value, attr_index):
    index = []
    for i in range(len(data.T[attr_index])):
        if data.T[attr_index][i] == value:
            index.append(i)
    return index

def accuracy(data, label, tree):
    correct = 0
    for i in range(data.shape[0]):
        if prediction(data[i], tree) == label[i]:
            correct += 1 
    return float(correct)/len(label)

def prediction(sample, tree):
    if isinstance(tree, Leaf): 
        return tree.classification
    else:
        idx = int(sample[tree.attr.attr_index])
        new_node = tree.subnode[idx]
    return prediction(sample, new_node)

def print_tree(tree):    
    if isinstance(tree, Node):
        print tree
        print_tree(tree.subnode[0])
        print_tree(tree.subnode[1])
 

input_tree = decision_tree_learning(train_data, train_label, attribute_list, 4, default=None)
acc_test = accuracy(test_data, test_label, input_tree) * 100
print_tree(input_tree)

# accuracy_test = []
# accuracy_train = []
# max_depth_list = range(22)
# for i in max_depth_list:
#     input_tree = decision_tree_learning(train_data, train_label, attribute_list, i, default=None)
#     acc_test = accuracy(test_data, test_label, input_tree) * 100
#     acc_train = accuracy(train_data, train_label, input_tree) * 100
#     accuracy_test.append(acc_test)
#     accuracy_train.append(acc_train)
#     print 'max_depth =', i, ': accuracy for test data is', acc_test
#     print 'max_depth =', i, ': accuracy for train data is', acc_train

# plt.plot(max_depth_list, accuracy_test, marker='o')
# plt.plot(max_depth_list, accuracy_train, marker='o')
# plt.legend(['test data', 'train data'], loc='upper left')
# plt.xlabel('max_depth')
# plt.ylabel('accuracy(%)')
# plt.show()
