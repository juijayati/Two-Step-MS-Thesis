import numpy as np
import os


superclass = 3
print( '##################################################################### for TRAIN #########################################################################')

tr_sample_size_per_class = 700
ts_sample_size_per_class = 50

destination_folder = 'DataSplit2_'
print('Deastination Folder: ',destination_folder)
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)
tr_features = np.load('DataSplit2/features_50ms.npy')
print(tr_features.shape)
tr_labels = np.load('DataSplit2/labels_50ms.npy')
print(tr_labels.shape)

tr = np.concatenate((tr_features,tr_labels), axis=1)
print(tr.shape)
#tr = np.delete(tr, np.s_[52:180:1], 1)
#print(tr.shape)
samples_per_class = np.sum(tr_labels, axis = 0)
print(samples_per_class)

#extract break_concrete
break_concrete = tr[np.where(tr[:,281]==1.0)]
tr_new = break_concrete[0:tr_sample_size_per_class,:]
print(tr_new.shape)

#extract excavating
excavating = tr[np.where(tr[:,282]==1.0)]
tr_new = np.concatenate((tr_new, excavating[0:tr_sample_size_per_class,:]), axis=0)
print(tr_new.shape)

#extract dozer
dozer = tr[np.where(tr[:,283]==1.0)]
tr_new = np.concatenate((tr_new, dozer[0:tr_sample_size_per_class,:]), axis=0)
print(tr_new.shape)


#extract dumper
dumper = tr[np.where(tr[:,285]==1.0)]
tr_new = np.concatenate((tr_new, dumper[0:tr_sample_size_per_class,:]), axis=0)
print(tr_new.shape)

#extract grader
grader = tr[np.where(tr[:,286]==1.0)]
tr_new = np.concatenate((tr_new, grader[0:tr_sample_size_per_class,:]), axis=0)
print(tr_new.shape)

#extract welding
welding = tr[np.where(tr[:,287]==1.0)]
tr_new = np.concatenate((tr_new, welding[0:tr_sample_size_per_class,:]), axis=0)
print(tr_new.shape)

#extract concrete_mixing
mixing = tr[np.where(tr[:,288]==1.0)]
tr_new = np.concatenate((tr_new, mixing[0:tr_sample_size_per_class,:]), axis=0)
print(tr_new.shape)

#extract concrete_pouring
pouring = tr[np.where(tr[:,289]==1.0)]
tr_new = np.concatenate((tr_new, pouring[0:tr_sample_size_per_class,:]), axis=0)
print(tr_new.shape)


#extract hammering
hammering = tr[np.where(tr[:,290]==1.0)]
tr_new = np.concatenate((tr_new, hammering[0:tr_sample_size_per_class,:]), axis=0)
print(tr_new.shape)

#extract grinding
grinding = tr[np.where(tr[:,291]==1.0)]
tr_new = np.concatenate((tr_new, grinding[0:tr_sample_size_per_class,:]), axis=0)
print(tr_new.shape)

#extract crane
crane = tr[np.where(tr[:,292]==1.0)]
tr_new = np.concatenate((tr_new, crane[0:tr_sample_size_per_class,:]), axis=0)
print(tr_new.shape)

#extract grinding
drilling = tr[np.where(tr[:,293]==1.0)]
tr_new = np.concatenate((tr_new, drilling[0:tr_sample_size_per_class,:]), axis=0)
print(tr_new.shape)

#extract nailing
nailing = tr[np.where(tr[:,294]==1.0)]
tr_new = np.concatenate((tr_new, nailing[0:tr_sample_size_per_class,:]), axis=0)
print(tr_new.shape)

tr_new_features = tr_new[:,0:281]
print(tr_new_features.shape)

tr_new_labels = tr_new[:,281:295]

print(tr_new_labels.shape)

tr_new_super_labels = np.zeros((tr_new_labels.shape[0],superclass))

tr_new_super_labels[0:(5*tr_sample_size_per_class),0] = 1.0
tr_new_super_labels[(5*tr_sample_size_per_class):(9*tr_sample_size_per_class),1] = 1.0
tr_new_super_labels[(9*tr_sample_size_per_class):,2] = 1.0


#np.save(os.path.join(destination_folder, 'tr_features.npy'), tr_new_features)
#np.save(os.path.join(destination_folder, 'tr_labels.npy'), tr_new_labels)
#np.save(os.path.join(destination_folder, 'tr_super_labels.npy'), tr_new_super_labels)



