import numpy as np
import math
import copy

depth = 0
## Decision Tree Learning
class Node():
    def __init__(self, attr, subnode):
        self.attr = attr
        self.subnode = subnode

class Leaf():
    def __init__(self, classfication):
        self.classfication = classfication
    
class Attribute():
    def __init__(self, name, values, index):
        self.name = name
        self.values = values
        self.index = index

def read_file(data_file, label_file, m):
    lines = [int(line.rstrip('\n')) for line in open(label_file)]
    label = np.array(lines)
    n = label.size

    data = np.zeros((n,m))
    with open(data_file, 'r') as in_file:
        for line in in_file:
            nums = line.split()
            data[int(nums[0])-1][int(nums[1])-1] = 1

    return (data, label)

attributes = []
i = -1
for line in open('words.txt'):
    i+=1
    attributes.append(Attribute(line, [0,1], i))
    

attribute_list = np.array(attributes)
m = len(attribute_list)
train_data, train_label = read_file('trainData.txt', 'trainLabel.txt', m)[0], read_file('trainData.txt', 'trainLabel.txt', m)[1]
test_data, test_label = read_file('testData.txt', 'testLabel.txt', m)[0], read_file('testData.txt', 'testLabel.txt', m)[1]

def decision_tree_learning(data, label, attribute_list, max_depth, default=None):
    global depth
    ## examples is empty
    if depth >= max_depth: return get_mode(label)
    depth += 1
    if data.shape[0] == 0:
        return Leaf(default)
    ## all examples have the same classification
    elif same_label(label):
        return Leaf(label[0])
    ## attribute is empty
    elif data.shape[1] == 0:
        return Leaf(get_mode(label))
    ## recursive decision tree learning
    else:
        best_attr_index = choose_attribute(data, label)
        print 'best_attr_index is ', attribute_list[best_attr_index].index
        mode = get_mode(label)
        ## delete the colum of best attribute
        data = np.delete(data, best_attr_index, axis=1)
        np.delete(attribute_list, [best_attr_index])
        subnode = []
        ## build subtree for every value of the attribute
        attr_values = attribute_list[best_attr_index].values
        for value in attr_values:
            index = split_data(data, value, best_attr_index)
            sub_data = data[index]
            sub_label = label[index]
            sub_attribute_list = attribute_list[index]
            subnode.append(decision_tree_learning(sub_data, sub_label, sub_attribute_list, max_depth, mode))
        return Node(attribute_list[best_attr_index], subnode)
        

def same_label(label):
    '''justify whether the label of all examples same'''
    for i in range(len(label)):
        if label[0] != label[i]:
            return False
    return True

def get_mode(label):
    '''return the majority label of all examples'''
    label_count = {}
    for i in label:
        if i not in label_count.keys(): 
            label_count[i] = 1
        else:
            label_count[i] += 1
    sorted_label_count = sorted(label_count, key=label_count.get, reverse=True)
    return sorted_label_count[0]
                
def choose_attribute(data, label):
    orig_entropy = calentropy(label)
    information_gain = []
    for x in range(np.shape(data)[1]):
        attr_values = attribute_list[x].values
        information_gain_temp = 0
        for value in attr_values:
            count_value = [i for i in data.T[x] if value == i]
            prob = len(count_value)/float(len(label))
            if prob != 0:
                index = split_data(data, value, x)
                new_data = data[index]
                new_label = label[index]
                new_entropy = calentropy(new_label)
                information_gain_temp += prob * new_entropy
        information_gain.append(orig_entropy - information_gain_temp)
    return information_gain.index(max(information_gain))

def calentropy(label):
    total = len(label)
    label_count = {}
    for i in label:
        if i not in label_count.keys(): 
            label_count[i] = 1
        else:
            label_count[i] += 1
    entropy = 0
    for x in label_count.values():
        entropy += ((-1)* float(x)/total * math.log(float(x)/total,2))
    return entropy
    
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
        idx = tree.attr.values.index(sample[tree.attr.index])
        new_node = tree.subnode[idx]
    return prediction(sample, new_node)
    
input_tree = decision_tree_learning(train_data, train_label, attribute_list, 1, default=None)
acc = accuracy(test_data, test_label, input_tree)
print acc

