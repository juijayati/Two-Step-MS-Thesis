#!/usr/bin/env python

import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import config as cf
import math


from scipy.signal import butter

import queue

def extract_feature(X, sample_rate):
    pos = 0;
    X = X.T
    #print(len(X))
    # short term fourier transform
    stft = np.abs(librosa.stft(X))

    # mfcc (mel-frequency cepstrum)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)

    pos += len(mfccs)

    print('Mfcc : ',pos)
    # chroma
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    #print(chroma.shape)
    # melspectrogram

    pos += len(chroma)

    print('Chroma : ',pos)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    #print(mel.shape)

    pos += len(mel)

    print('Mel : ',pos)
    # spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    #print(contrast.shape)

    pos += len(contrast)

    print('Contrast : ',pos)

    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)

    pos += len(tonnetz)

    print('Tonnetz : ',pos)

    #print(tonnetz.shape)
    #zero crossings
    zcc = np.array(librosa.feature.zero_crossing_rate(y = X,frame_length= math.ceil(cf.frame_zcc / cf.period)).T)
    zcc.shape = (zcc.shape[0],)
    print(zcc)

    pos += len(zcc)

    print('Zcc : ',pos)


    #var zcc
    vzcr = np.array([np.var(zcc, ddof = 1)])

    pos += len(vzcr)
    print('Vzcr : ', pos)


    #print(vzcr.shape)
    #HZCRR
    av_zcc = np.mean(zcc)


    #sign_zcc = np.subtract(X)
    sign_zcc = np.sign(np.subtract(zcc, 1.5 * av_zcc))
    hzcrr = np.array([sum(np.add(sign_zcc , 1) / (2*len(zcc)))])

    pos += 1

    print('Hzccr : ',pos)

    #rms
    rms = np.array(librosa.feature.rmse(y = X, frame_length = math.ceil(cf.frame_rms / cf.period)).T)
    rms.shape = (rms.shape[0],)

    pos += len(rms)
    print('Rms : ',pos)

    #Low Energy Frame
    av_rms = np.mean(rms)
    lef = np.array([sum(rms < av_rms * .5) / len(rms)])

    pos += len(lef)

    print('Lef : ',pos)


    #Short Time Energy
    ste_samples = math.ceil(cf.frame_ste / cf.period)
    ste = np.array(ste_calculator(X, ste_samples))

    pos += len(ste)
    print('Ste : ',pos)


    #Low Short Time Energy Ratio
    av_ste = np.mean(ste)
    sign_ste = np.sign(np.subtract(0.5 * av_ste, ste))
    lster = np.array([sum(np.add(sign_ste , 1) / (2 * len(ste)))])

    pos += len(lster)
    print('Lster : ',pos)


    #variance in spectrum flux
    vsf_samples = math.ceil(cf.frame_vsflux / cf.period)
    vsf = np.array(spectral_flux_calculator(X, vsf_samples))
    vsflux = np.array([np.var(vsf, ddof = 1)])

    pos += len(vsflux)
    print('Vsfflux : ',pos)


    #High-Order Crossing
    # print(len(X))
    data_ho = np.zeros((len(X),4))
    data_ho[:,0] = np.append( np.diff(X, n=1), np.zeros(1))
    data_ho[:,1] = np.append( np.diff(X, n=2), np.zeros(2))
    data_ho[:, 2] = np.append(np.diff(X, n=3), np.zeros(3))
    data_ho[:, 3] = np.append(np.diff(X, n=4), np.zeros(4))
    vhoc = np.zeros(len(data_ho[0]))
    mhoc = np.zeros(len(data_ho[0]))

    for k in range(len(data_ho[0])):
        ZHOC = np.array(librosa.feature.zero_crossing_rate(y = data_ho[:,k].T, frame_length= math.ceil(cf.frame_zcc / cf.period)).T)
        vhoc[k] = np.var(ZHOC, ddof = 1)
        mhoc[k] = np.mean(ZHOC)

    # LPC Co-efficients

    pos += len(vhoc)
    print('Vhoc : ',pos)

    pos += len(mhoc)
    print('Mhoc : ',pos)





    return mfccs,chroma,mel,contrast,tonnetz,zcc,rms,vzcr,hzcrr,lef,ste,lster,vsflux,vhoc,mhoc


def ste_calculator(data, frame_s, overlap_s = 0):
    STE_frames = frame_extractor(data, frame_s)
    STE = np.zeros(len(STE_frames[0]))
    for i in range(len(STE)):
        STE[i] = sum(STE_frames[:,i]**2)
    return STE

def ler_calculator(data, frame_s, overlap_s = 0):
    LE_frames = frame_extractor(data, frame_s)


def spectral_flux_calculator(data, frame_s, overlap_s = 0):
    VSF_frames = frame_extractor(data, frame_s)
    VSF = np.zeros(len(VSF_frames[0]),dtype=complex)

    #fft size
    NFFT = 2 ** math.ceil(math.log2(len(VSF_frames)))
    sp_prev = np.zeros(NFFT)
    sp_nopd = len(VSF_frames)

    for i in range(len(VSF_frames[0])):
        #log less algorithm
        sp_curr = np.fft.fft(VSF_frames[:,i], NFFT) / sp_nopd
        sp_temp = np.subtract(sp_curr, sp_prev) ** 2
        VSF[i] = sum(sp_temp)
        sp_prev = sp_curr
    return VSF

def silence_detector(signal, frame_samples, overlap_samples = 0):
    XS = len(signal)
    N = frame_samples
    M = overlap_samples
    L = N - M
    F = math.floor((XS - M) / L)
    data = np.empty(0)
    # print(frames.shape)
    for n in range(F):
        a = n * L
        b = (n + 1) * L
        frame = np.array(signal[a:b])
        # print(frame.shape)
        frame_squared_sum = frame ** 2
        sd_rms = 20 * math.log10(math.sqrt(sum(frame_squared_sum) / N))
        if (sd_rms > cf.thres_sd):
            data = np.append(data, frame)
    #print('non silent data ', data.shape)

    return data




def frame_extractor(signal, frame_samples, overlap_samples = 0):
    XS = len(signal)
    N = frame_samples
    M = overlap_samples
    L = N - M
    F = math.floor((XS - M)/L)
    frames = np.zeros((N,F))
    # print(frames.shape)
    for n in range(F):
        a = n * L
        b = (n + 1) * L
        frames[:,n] = np.array(signal[a:b])

    #print('super frames ',frames.shape)

    return frames




def parse_audio_files(parent_dir, file_ext='*.wav'):
    sub_dirs = os.listdir(parent_dir)
    sub_dirs.sort()
    print(sub_dirs)
    features, labels = np.empty((0,cf.num_features)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        if os.path.isdir(os.path.join(parent_dir, sub_dir)):
            label_name = int(sub_dir.split('_')[0]) - 1
            print('here ', sub_dir)
            num_frames_per_class = 0
            limit = False
            #print('label ',label_name)
            for file_name in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
                if file_name and ~limit:
                    try:
                        #print('Extracting', file_name)
                        X, sample_rate = sf.read(file_name, dtype='float32')
                        print('sample length ',len(X))
                        cf.period = 1.0 / sample_rate
                        print('period : ',cf.period)
                        if X.ndim > 1: X = X[:, 0]
                        #print(X)
                        X = np.divide(X, max(X))
                        #print('data: ',X)
                        sd_samples = math.ceil(cf.frame_sd / cf.period)
                        sf_samples = math.ceil(cf.super_frame / cf.period)
                        X = silence_detector(X, sd_samples)
                        frames = frame_extractor(X, sf_samples)
                        num_frames = len(frames[0])
                        for i in range(num_frames):

                            frame_i = frames[:,i]
                            #print(frame_i.shape)
                            mfccs, chroma, mel, contrast,tonnetz, zcc, rms, vzcr, hzccr, lef, ste, lster, vsflux, vhoc, mhoc = extract_feature(frame_i, sample_rate)
                            ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz, zcc, rms, vzcr, hzccr, lef, ste, lster,vsflux, vhoc, mhoc])
                            #print(ext_features.shape)
                            features = np.vstack([features, ext_features])
                            # c = '\\';
                            # label = file_name.split(c)[1]
                            labels = np.append(labels, label_name)
                            num_frames_per_class += 1
                            #break
                            print(num_frames_per_class)

                            if num_frames_per_class == 3000:
                                print('Train Size after Label ', label_name, features.shape[0])
                                limit = True
                                break
                            # print("extract %s features done" % (sub_dir))
                        if limit:
                            break
                    except Exception as e:
                       print("[Error] extract feature error in %s. %s" % (file_name,e))
                       #continue

                #break

    return np.array(features), np.array(labels, dtype=np.int)

def get_classes(parent_dir):
    return os.listdir(parent_dir)

def encode_labels(labels):

    n_labels = len(labels)
    one_encode = np.zeros((n_labels, cf.num_classes))
    one_encode[np.arange(n_labels), labels] = 1
    print(one_encode)
    return one_encode


def parse_audio_files_2(parent_dir, file_ext='*.wav'):
    sub_dirs = os.listdir(parent_dir)
    sub_dirs.sort()
    print(sub_dirs)
    features, labels = np.empty((0,cf.num_features)), np.empty(0)
    avg, std = np.empty(0), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        if os.path.isdir(os.path.join(parent_dir, sub_dir)):
            label_name = int(sub_dir.split('_')[0]) - 1
            print('here ', sub_dir)
            num_frames_per_class = 0
            limit = False
            #print('label ',label_name)
            for file_name in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
                if file_name and ~limit:
                    try:
                        #print('Extracting', file_name)
                        X, sample_rate = sf.read(file_name, dtype='float32')
                        #print("sample length : %d    sample_rate : %f " % (len(X),sample_rate))
                        cf.period = 1.0 / sample_rate
                        #print('period : ',cf.period)

                        if X.ndim > 1: X = X[:, 0]
                        #print(X)
                        X = np.divide(X, max(X))
                        #print('data: ',X)
                        #sd_samples = math.ceil(cf.frame_sd / cf.period)
                        sf_samples = math.ceil(cf.super_frame / cf.period)
                        #X = silence_detector(X, sd_samples)
                        frames = frame_extractor(X, sf_samples)
                        num_frames = len(frames[0])
                        for i in range(num_frames):
                            frame_i = frames[:,i]
                            avg = np.append(avg, np.average(frame_i))
                            std = np.append(std, np.std(frame_i))

                            num_frames_per_class += 1
                            # break
                            #print(num_frames_per_class)

                            if num_frames_per_class == 300:
                                #print('Train Size after Label ', label_name, features.shape[0])
                                limit = True
                                break
                            # print("extract %s features done" % (sub_dir))
                        if limit:
                            break
                    except Exception as e:
                       print("[Error] extract feature error in %s. %s" % (file_name,e))
                       #continue

                #break

    return avg, std




def main():

    cf.num_classes = len(get_classes('DataSplit\Train'))

    features = np.load('DataSplit2/features_200ms.npy')
    labels = np.load('DataSplit2/labels_200ms.npy')

    '''current_sample_count = int(np.sum(labels[:,8]))
    print(current_sample_count)

    itemindex = np.where(labels[:,8]==1.0)
    lastindex = np.max(itemindex)

    print(lastindex)


    tr_features, tr_labels = parse_audio_files_2('DataSplit\Test', current_sample_count)
    tr_labels = encode_labels(tr_labels)
    print(tr_features.shape)
    features = np.vstack([features,tr_features])
    labels = np.vstack([labels, tr_labels])
    print(features.shape, labels.shape)
    #np.save(os.path.join('DataSplit2', 'features_50ms.npy'), features)
    #np.save(os.path.join('DataSplit2', 'labels_50ms.npy'), labels)
    #ts_features, ts_labels = parse_audio_files('DataSplit\Test')
    #print(ts_labels)
    #ts_labels = encode_labels(ts_labels)
    #np.save('ts_features4_split.npy', ts_features)
    #np.save('ts_labels4_split.npy', ts_labels)

    # Predict new'''

    avg, std = parse_audio_files_2('DataSplit\Train')
    print(std)
    print(avg)

    plt.plot(std,avg,'ro')
    plt.axis([-0.2, 0.5,-0.04,0.05])
    plt.show()


if __name__ == '__main__': main()