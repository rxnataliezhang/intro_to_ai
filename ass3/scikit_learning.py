from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB
import numpy as np

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

attribute_list = [line for line in open('words.txt')]
m = len(attribute_list)
train_data, train_label = read_file('trainData.txt', 'trainLabel.txt', m)[0], read_file('trainData.txt', 'trainLabel.txt', m)[1]
test_data, test_label = read_file('testData.txt', 'testLabel.txt', m)[0], read_file('testData.txt', 'testLabel.txt', m)[1]

# scikit-learn for test

clf = DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=5)
clf.fit(train_data, train_label)
pred = clf.predict(test_data)
print "Accuracy:", accuracy_score(test_label, pred)

# clf = BernoulliNB(alpha=1,binarize=None)
# clf.fit(train_data, train_label)
# pred = clf.predict(test_data)
# print "Accuracy:", accuracy_score(test_label, pred)
