import numpy as np
import matplotlib.pyplot as plt

tr_features = np.load('DataSplit_all_all/tr_features.npy')

print(tr_features.shape)



dist = np.zeros((tr_features.shape[0]*(tr_features.shape[0]-1)))
index = 0

for i in range (tr_features.shape[0]):
    for j in range (tr_features.shape[0]):
        if i != j :
            dist[index]  = np.linalg.norm(tr_features[i]-tr_features[j])
            index += 1
middle = len(dist) / 2
plt.plot(dist[0:int(middle)])