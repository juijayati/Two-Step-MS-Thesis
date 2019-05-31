period = 0.0000001

super_frame = 0.050     # superframe (sec)

# slicence_detector
frame_sd = 0.050           # silence detector subframe (sec)
overlap_sd = 0             # silence detector overlap (sec)
thres_sd = -30             # silence detector RMS threshold (dB)

# feature_extractor
frame_zcc = 0.005          # zero-crossing count subframe (sec)
overlap_zcc = 0.0025       # zero-crossing count overlap (sec)

frame_rms = 0.005          # root-mean-square subframe (sec)
overlap_rms = 0.0025       # root-mean-square overlap (sec)

frame_ste = 0.005          # short time energy subframe (sec)
overlap_ste = 0.0025       # short time energy overlap (sec)

frame_vler = 0.020     #0.020     # variance low energy subframe (sec)
overlap_vler = 0.010   #0.010     # variance low energy overlap (sec)

frame_vsflux = 0.020   #0.025      # variance spectral flux subframe (sec)
overlap_vsflux = 0.010 #0.015      # variance spectral flux overlap (sec)

frame_lfrms = 0.010        # low frequency RMS subframe (sec)
overlap_lfrms = 0.005      # low frequency RMS overlap (sec)

frame_zcc9 = 0.010         # variance 9th order zcc subframe (sec)
overlap_zcc9 = 0.005       # variance 9th order zcc overlap (sec)

frame_sbc = 0.015 #0.010  # sub-band correlation subframe (sec)
overlap_sbc = 0            # sub-band correlation overlap (sec)

lpc_order = 8              # Linear Prediction Coefficients

num_features = 393
num_classes = 1

train_dir = 'DataSplit2\Train'
test_dir = 'DataSplit2\Test'


# num_features = 243, num_samples_per_class = 1500 for 100 ms
# num_features = 261, num_samples_per_class = 1000 for 150 ms
# num_features = 281, num_samples_per_class = 750 for 200 ms

# num_features = 299, num_samples_per_class = 600 for 250 ms
# num_features = 299, num_samples_per_class = 600 for 350 ms


# num_features = 225 for 50 ms
# num_features = 317 for 300 ms
# num_features = 355 for 400 ms
# num_features = 393 for 500 ms
# num_features = 579 for 1000 ms
# num_features = 2065 for 5000 ms

