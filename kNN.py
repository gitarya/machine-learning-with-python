import numpy as np
import operator
from collections import Counter

def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def kNN(test, train, labels, k):
    num = train.shape[0]
    diff = np.tile(test, (num,1)) - train
    sqDiff = diff**2
    distance = sqDiff.sum(axis=1) ** 0.5
    sortDisIndex = distance.argsort()
    classCount = []
    for i in range(k):
        classCount.append(labels[sortDisIndex[i]])
    return list(Counter(classCount).items())[0][0]

group, labels = createDataSet()
print(kNN([0,0], group, labels, 3))
