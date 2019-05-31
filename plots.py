import matplotlib.pyplot as plt
import os

destination_folder = 'DataSplit3(random)_700_50/Photos'



#labels = ['50', '100','200','(','(700,50)']


train = [50, 100, 200, 250, 500, 1000, 2000]
train_label = [50, 100, 200, 250, 500, 1000, 5000]

#n_cluster = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
#total_ssw = [77008228,62163656,52767651,45468636,39094454,35394611,32550468, 30708558, 29081507, 27780576]

#gen = [0.8992, 0.9076, 0.9081, 0.9071, 0.8986, 0.8489, 0.1333]
#hier = [0.9452, 0.9510, 0.9512, 0.9426, 0.9167, 0.6661, 0.1422]

frame_size = [50, 100, 200, 250, 500, 1000, 2000]
gen = [0.8992, 0.9076, 0.9081, 0.9071, 0.8986, 0.8489, 0.1333]
hier = [0.9452, 0.9510, 0.9515, 0.9426, 0.9167, 0.6661, 0.1422]


n_cluster = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
k_means = [0.5330, 0.6152, 0.7126, 0.7643, 0.7901, 0.7931, 0.8304, 0.8494, 0.8721, 0.8789]
agglo = [0.5397, 0.6094, 0.6941, 0.7145, 0.7780, 0.7883, 0.8021, 0.8567, 0.8746, 0.8855]


#plt.plot(train, gen, 'sr-', train, hier, 'xb-')
plt.plot(frame_size, gen, 'sr-', frame_size, hier, 'xb-')
#plt.tight_layout()
#plt.gcf().subplots_adjust(bottom=0.15)    # use this if the xlabel is cut from the saved file

plt.xlabel("Frame Size (ms)")
plt.ylabel("Accuracy")
plt.xticks(train,train_label, rotation = 55)
plt.axis([50, 2000, 0.0, 1.0])
plt.gca().legend(('Simple Neural Classification','Two-Step Neural Classification'))
plt.savefig(os.path.join(destination_folder,'Iteration_frame_size_2.png'))


plt.show()



