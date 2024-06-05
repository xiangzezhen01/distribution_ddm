from skmultiflow.data import SEAGenerator
from skmultiflow.trees import HoeffdingTree
from skmultiflow import EvaluatePrequential
import matplotlib.pyplot as plt
import numpy as np

stream = SEAGenerator()
'''创建数据流
stream.prepare_for_use()
nb_iters=100
tree = HoeffdingTree() '''

correctness_dist = []

for i in range(nb_iters):
    X, Y = stream.next_sample()
    prediction = tree.predict(X)
    if Y == prediction:
        correctness_dist.append(1)
    else:
        correctness_dist.append(0)
    tree.partial_fit(X, Y)

time = [i for i in range(nb_iters)]
sumValue = 0
accuracy = np.zeros(len(correctness_dist))
for i in range(len(correctness_dist)):
    sumValue = sumValue + correctness_dist[i]
    accuracy[i] = (sumValue * 1.0) / (i + 1)

print(accuracy)
plt.plot(time, accuracy)
