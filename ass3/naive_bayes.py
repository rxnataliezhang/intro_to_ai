import numpy as np
import math

attribute_list = [line.strip('\n') for line in open('words.txt')]
m = len(attribute_list)

def read_file(data_file, label_file):
    lines = [int(line.rstrip('\n')) for line in open(label_file)]
    label = np.array(lines)
    n = label.size

    data = np.zeros((n,m))
    with open(data_file, 'r') as in_file:
        for line in in_file:
            nums = line.split()
            data[int(nums[0])-1][int(nums[1])-1] = 1
    return (data, label)

train_data, train_label = read_file('trainData.txt', 'trainLabel.txt')[0], read_file('trainData.txt', 'trainLabel.txt')[1]
test_data, test_label = read_file('testData.txt', 'testLabel.txt')[0], read_file('testData.txt', 'testLabel.txt')[1]

def naive_bayes_model(data, label):
    c1_index = [i for i in range(len(label)) if label[i] == 1]
    c2_index = [i for i in range(len(label)) if label[i] == 2]
    theta = float(len(c1_index)+1) / (len(label)+2)

    theta_c1 = []
    for word in data[c1_index].T:
        exist_word = [i for i in word if i == 1]
        theta_c1.append(float(len(exist_word)+1)/(len(word)+2))

    theta_c2 = []
    for word in data[c2_index].T:
        exist_word = [i for i in word if i == 1]
        theta_c2.append(float(len(exist_word)+1)/(len(word)+2)) 

    return (theta, theta_c1, theta_c2)
        
def prediction(data, label, theta, theta_c1, theta_c2):
    correct = 0
    predict = []
    for j in range(data.shape[0]):
        sample = data[j]
        sample_theta_c1 = []
        sample_theta_c2 = []
        for i in range(len(sample)):
            if sample[i] == 1: 
                sample_theta_c1.append(theta_c1[i])
                sample_theta_c2.append(theta_c2[i])
            else:
                sample_theta_c1.append(1 - theta_c1[i])
                sample_theta_c2.append(1 - theta_c2[i])
        sample_theta_c1 = np.array(sample_theta_c1)
        sample_theta_c2 = np.array(sample_theta_c2)
        prob_c1 = np.prod(sample_theta_c1) * theta
        prob_c2 = np.prod(sample_theta_c2) * (1-theta)
        if prob_c1 > prob_c2: pred = 1
        else: pred = 2
        predict.append(pred)
        if label[j] == pred: correct += 1
        accuracy = float(correct)/len(label)
    return (predict, accuracy)
            
theta, theta_c1, theta_c2 = naive_bayes_model(train_data, train_label)
accuracy_test = prediction(test_data, test_label, theta, theta_c1, theta_c2)[1]
pred_test = prediction(test_data, test_label, theta, theta_c1, theta_c2)[0]
accuracy_train = prediction(train_data, train_label, theta, theta_c1, theta_c2)[1]
pred_train = prediction(train_data, train_label, theta, theta_c1, theta_c2)[0]
print 'trainig accuracy is ', accuracy_train
print 'test accuracy is ', accuracy_test

# theta_c1 = np.array(theta_c1)
# theta_c2 = np.array(theta_c2)
# theta_c1 = np.log10(theta_c1)
# theta_c2 = np.log10(theta_c2)
# disc = theta_c1 - theta_c2
# disc = np.absolute(disc)
# sort_disc = sorted(range(len(disc)), key=lambda k: disc[k], reverse=True)[0:10]
# attribute_list = np.array(attribute_list)
# print attribute_list[sort_disc]
# print disc[sort_disc]



