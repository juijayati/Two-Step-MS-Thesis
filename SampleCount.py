import os

tr_dir = 'DataSplit\Train'
ts_dir = 'DataSplit\Test'

file_ext = '*.wav'
sub_dirs_ts = os.listdir(ts_dir)
sub_dirs_tr = os.listdir(tr_dir)

sub_dirs_tr.sort()
sub_dirs_ts.sort()

print('Class \t\t # Train \t\t # Test')

for sub_dir_tr, sub_dir_ts in zip(sub_dirs_tr, sub_dirs_ts):
    class_name = sub_dir_tr
    sub_dir_tr_path = os.path.join(tr_dir, sub_dir_tr)
    sub_dir_ts_path = os.path.join(ts_dir, sub_dir_ts)
    num_samples_tr = len([name for name in os.listdir(sub_dir_tr_path) if os.path.isfile(os.path.join(sub_dir_tr_path, name))])
    num_samples_ts = len([name for name in os.listdir(sub_dir_ts_path) if os.path.isfile(os.path.join(sub_dir_ts_path, name))])
    print("%s \t\t %s \t\t %s" % (class_name, num_samples_tr, num_samples_ts))
