import numpy as np
import os
np.random.seed(0)

superclass = 3
print('##################################################################### for TRAIN #########################################################################')

tr_sample_size_per_class = 700
ts_sample_size_per_class = 50
total_sample_size_per_class = tr_sample_size_per_class + ts_sample_size_per_class

destination_folder = 'DataSplit2(random)_' + str(tr_sample_size_per_class) + '_' + str(ts_sample_size_per_class)
print('Deastination Folder: ',destination_folder)
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)
tr_features = np.load('DataSplit_all_all/tr_features.npy')
print(tr_features.shape)
tr_labels = np.load('DataSplit_all_all/tr_labels.npy')
print(tr_labels.shape)


tr = np.concatenate((tr_features,tr_labels), axis=1)
print(tr.shape)
#tr = np.delete(tr, np.s_[52:180:1], 1)
#print(tr.shape)
samples_per_class = np.sum(tr_labels, axis = 0).astype(int)
print(samples_per_class)


#extract break_concrete

ts_sample_indices = np.random.randint(total_sample_size_per_class, size = ts_sample_size_per_class)
tr_sample_indices = np.arange(total_sample_size_per_class)
tr_sample_indices = np.delete(tr_sample_indices, ts_sample_indices)

break_concrete = tr[np.where(tr[:,281]==1.0)]
break_concrete = break_concrete[0:total_sample_size_per_class,:]
tr_new = break_concrete[tr_sample_indices]
ts_new = break_concrete[ts_sample_indices]

print(tr_new.shape)
print(ts_new.shape)

#extract excavating

ts_sample_indices = np.random.randint(total_sample_size_per_class, size = ts_sample_size_per_class)
tr_sample_indices = np.arange(total_sample_size_per_class)
tr_sample_indices = np.delete(tr_sample_indices, ts_sample_indices)

excavating = tr[np.where(tr[:,282]==1.0)]
excavating = excavating[0:total_sample_size_per_class,:]
tr_new = np.concatenate((tr_new, excavating[tr_sample_indices]), axis = 0)
ts_new = np.concatenate((ts_new, excavating[ts_sample_indices]), axis = 0)

print(tr_new.shape)
print(ts_new.shape)

#extract dozer

ts_sample_indices = np.random.randint(total_sample_size_per_class, size = ts_sample_size_per_class)
tr_sample_indices = np.arange(total_sample_size_per_class)
tr_sample_indices = np.delete(tr_sample_indices, ts_sample_indices)

dozer = tr[np.where(tr[:,283]==1.0)]
dozer = dozer[0:total_sample_size_per_class,:]
tr_new = np.concatenate((tr_new, dozer[tr_sample_indices]), axis = 0)
ts_new = np.concatenate((ts_new, dozer[ts_sample_indices]), axis = 0)

print(tr_new.shape)
print(ts_new.shape)

#extract dumper

ts_sample_indices = np.random.randint(total_sample_size_per_class, size = ts_sample_size_per_class)
tr_sample_indices = np.arange(total_sample_size_per_class)
tr_sample_indices = np.delete(tr_sample_indices, ts_sample_indices)

dumper = tr[np.where(tr[:,285]==1.0)]
dumper = dumper[0:total_sample_size_per_class,:]
tr_new = np.concatenate((tr_new, dumper[tr_sample_indices]), axis = 0)
ts_new = np.concatenate((ts_new, dumper[ts_sample_indices]), axis = 0)

print(tr_new.shape)
print(ts_new.shape)

#extract grader

ts_sample_indices = np.random.randint(total_sample_size_per_class, size = ts_sample_size_per_class)
tr_sample_indices = np.arange(total_sample_size_per_class)
tr_sample_indices = np.delete(tr_sample_indices, ts_sample_indices)

grader = tr[np.where(tr[:,286]==1.0)]
grader = grader[0:total_sample_size_per_class,:]
tr_new = np.concatenate((tr_new, grader[tr_sample_indices]), axis = 0)
ts_new = np.concatenate((ts_new, grader[ts_sample_indices]), axis = 0)

print(tr_new.shape)
print(ts_new.shape)

#extract welding
ts_sample_indices = np.random.randint(total_sample_size_per_class, size = ts_sample_size_per_class)
tr_sample_indices = np.arange(total_sample_size_per_class)
tr_sample_indices = np.delete(tr_sample_indices, ts_sample_indices)

welding = tr[np.where(tr[:,287]==1.0)]
welding = welding[0:total_sample_size_per_class,:]
tr_new = np.concatenate((tr_new, welding[tr_sample_indices]), axis = 0)
ts_new = np.concatenate((ts_new, welding[ts_sample_indices]), axis = 0)

print(tr_new.shape)
print(ts_new.shape)

#extract concrete_mixing

ts_sample_indices = np.random.randint(total_sample_size_per_class, size = ts_sample_size_per_class)
tr_sample_indices = np.arange(total_sample_size_per_class)
tr_sample_indices = np.delete(tr_sample_indices, ts_sample_indices)

mixing = tr[np.where(tr[:,288]==1.0)]
mixing = mixing[0:total_sample_size_per_class,:]
tr_new = np.concatenate((tr_new, mixing[tr_sample_indices]), axis = 0)
ts_new = np.concatenate((ts_new, mixing[ts_sample_indices]), axis = 0)

print(tr_new.shape)
print(ts_new.shape)

#extract concrete_pouring
ts_sample_indices = np.random.randint(total_sample_size_per_class, size = ts_sample_size_per_class)
tr_sample_indices = np.arange(total_sample_size_per_class)
tr_sample_indices = np.delete(tr_sample_indices, ts_sample_indices)

pouring = tr[np.where(tr[:,289]==1.0)]
pouring = pouring[0:total_sample_size_per_class,:]
tr_new = np.concatenate((tr_new, pouring[tr_sample_indices]), axis = 0)
ts_new = np.concatenate((ts_new, pouring[ts_sample_indices]), axis = 0)

print(tr_new.shape)
print(ts_new.shape)


#extract hammering
ts_sample_indices = np.random.randint(samples_per_class[9], size = ts_sample_size_per_class)
tr_sample_indices = np.arange(samples_per_class[9])
print(tr_sample_indices)
tr_sample_indices = np.delete(tr_sample_indices, ts_sample_indices)

hammering = tr[np.where(tr[:,290]==1.0)]
hammering = hammering[0:(len(tr_sample_indices)+len(ts_sample_indices)),:]
tr_new = np.concatenate((tr_new, hammering[tr_sample_indices]), axis = 0)
ts_new = np.concatenate((ts_new, hammering[ts_sample_indices]), axis = 0)

print(tr_new.shape)
print(ts_new.shape)

#extract grinding

ts_sample_indices = np.random.randint(total_sample_size_per_class, size = ts_sample_size_per_class)
tr_sample_indices = np.arange(total_sample_size_per_class)
tr_sample_indices = np.delete(tr_sample_indices, ts_sample_indices)

grinding = tr[np.where(tr[:,291]==1.0)]
grinding = grinding[0:total_sample_size_per_class,:]
tr_new = np.concatenate((tr_new, grinding[tr_sample_indices]), axis = 0)
ts_new = np.concatenate((ts_new, grinding[ts_sample_indices]), axis = 0)

print(tr_new.shape)
print(ts_new.shape)

#extract crane

ts_sample_indices = np.random.randint(total_sample_size_per_class, size = ts_sample_size_per_class)
tr_sample_indices = np.arange(total_sample_size_per_class)
tr_sample_indices = np.delete(tr_sample_indices, ts_sample_indices)

crane = tr[np.where(tr[:,292]==1.0)]
crane = crane[0:total_sample_size_per_class,:]
tr_new = np.concatenate((tr_new, crane[tr_sample_indices]), axis = 0)
ts_new = np.concatenate((ts_new, crane[ts_sample_indices]), axis = 0)

print(tr_new.shape)
print(ts_new.shape)
#extract drilling

ts_sample_indices = np.random.randint(total_sample_size_per_class, size = ts_sample_size_per_class)
tr_sample_indices = np.arange(total_sample_size_per_class)
tr_sample_indices = np.delete(tr_sample_indices, ts_sample_indices)

drill = tr[np.where(tr[:,293]==1.0)]
drill = drill[0:total_sample_size_per_class,:]
tr_new = np.concatenate((tr_new, drill[tr_sample_indices]), axis = 0)
ts_new = np.concatenate((ts_new, drill[ts_sample_indices]), axis = 0)

print(tr_new.shape)
print(ts_new.shape)



#extract nailing

ts_sample_indices = np.random.randint(samples_per_class[13], size = 5)
tr_sample_indices = np.arange(samples_per_class[13])
tr_sample_indices = np.delete(tr_sample_indices, ts_sample_indices)

nailing = tr[np.where(tr[:,294]==1.0)]
nailing = nailing[0:(len(tr_sample_indices)+len(ts_sample_indices)),:]
tr_new = np.concatenate((tr_new, nailing[tr_sample_indices]), axis = 0)
ts_new = np.concatenate((ts_new, nailing[ts_sample_indices]), axis = 0)

print(tr_new.shape)
print(ts_new.shape)

tr_new = np.reshape(tr_new,(tr_new.shape[0],tr.shape[1]))


tr_new_features = tr_new[:,0:281]
print(tr_new_features.shape)

ts_new_features = ts_new[:,0:281]
print(ts_new_features.shape)

tr_new_labels = tr_new[:,281:295]

print(tr_new_labels.shape)

tr_new_labels = np.delete(tr_new_labels, 3, 1)

print(tr_new_labels.shape)

ts_new_labels = ts_new[:,281:295]
print(ts_new_labels.shape)

ts_new_labels = np.delete(ts_new_labels, 3, 1)

print(ts_new_labels.shape)


tr_new_super_labels = np.zeros((tr_new_labels.shape[0],superclass))

tr_new_super_labels[0:(5*tr_sample_size_per_class),0] = 1.0
tr_new_super_labels[(5*tr_sample_size_per_class):(9*tr_sample_size_per_class),1] = 1.0
tr_new_super_labels[(9*tr_sample_size_per_class):,2] = 1.0


np.save(os.path.join(destination_folder, 'tr_features.npy'), tr_new_features)
np.save(os.path.join(destination_folder, 'tr_labels.npy'), tr_new_labels)
np.save(os.path.join(destination_folder, 'tr_super_labels.npy'), tr_new_super_labels)


ts_new_super_labels = np.zeros((ts_new_labels.shape[0],superclass))

ts_new_super_labels[0:(5*ts_sample_size_per_class),0] = 1.0
ts_new_super_labels[(5*ts_sample_size_per_class):(9*ts_sample_size_per_class),1] = 1.0
ts_new_super_labels[(9*ts_sample_size_per_class):,2] = 1.0

np.save(os.path.join(destination_folder, 'ts_features.npy'), ts_new_features)
np.save(os.path.join(destination_folder, 'ts_labels.npy'), ts_new_labels)
np.save(os.path.join(destination_folder, 'ts_super_labels.npy'), ts_new_super_labels)




