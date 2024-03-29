import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
import math
import contextlib
import os
import glob


cutOffFrequency = 400.0

# from http://stackoverflow.com/questions/13728392/moving-average-or-running-mean

def running_mean(x, windowSize):
  cumsum = np.cumsum(np.insert(x, 0, 0))
  return (cumsum[windowSize:] - cumsum[:-windowSize]) / windowSize

# from http://stackoverflow.com/questions/2226853/interpreting-wav-data/2227174#2227174
def interpret_wav(raw_bytes, n_frames, n_channels, sample_width, interleaved = True):

    if sample_width == 1:
        dtype = np.uint8 # unsigned char
    elif sample_width == 2:
        dtype = np.int16 # signed 2-byte short
    else:
        raise ValueError("Only supports 8 and 16 bit audio formats.")

    channels = np.fromstring(raw_bytes, dtype=dtype)

    if interleaved:
        # channels are interleaved, i.e. sample N of channel M follows sample N of channel M-1 in raw data
        channels.shape = (n_frames, n_channels)
        channels = channels.T
    else:
        # channels are not interleaved. All samples from channel M occur before all samples from channel M-1
        channels.shape = (n_channels, n_frames)

    return channels


def Denoise_2(fname,outname):
    with contextlib.closing(wave.open(fname,'rb')) as spf:
        sampleRate = spf.getframerate()
        ampWidth = spf.getsampwidth()
        nChannels = spf.getnchannels()
        nFrames = spf.getnframes()

        # Extract Raw Audio from multi-channel Wav File
        signal = spf.readframes(nFrames*nChannels)
        spf.close()
        channels = interpret_wav(signal, nFrames, nChannels, ampWidth, True)

        # get window size
        # from http://dsp.stackexchange.com/questions/9966/what-is-the-cut-off-frequency-of-a-moving-average-filter
        freqRatio = (cutOffFrequency/sampleRate)
        N = int(math.sqrt(0.196196 + freqRatio**2)/freqRatio)

        # Use moviung average (only on first channel)
        filtered = running_mean(channels[0], N).astype(channels.dtype)

        wav_file = wave.open(outname, "w")
        wav_file.setparams((1, ampWidth, sampleRate, nFrames, spf.getcomptype(), spf.getcompname()))
        wav_file.writeframes(filtered.tobytes('C'))
        wav_file.close()

def main():
    indir = 'DataSplit2\Train'
    outdir = 'DataSplit2Denoise\Train'

    file_ext = '*.wav'
    sub_dirs = os.listdir(indir)

    sub_dirs.sort()

    print(sub_dirs)

    for label, sub_dir in enumerate(sub_dirs):
        if (os.path.isdir(os.path.join(indir, sub_dir))):
            for file_name in glob.glob(os.path.join(indir, sub_dir, file_ext)):
                try:
                    if file_name:
                        print('Denoising file: ', file_name)
                        out_file_name = file_name.split('\\')[3]

                        out_subdir = outdir + '\\' + sub_dir + '\\'

                        if not os.path.exists(out_subdir):
                            os.makedirs(out_subdir)

                        out_file = out_subdir + out_file_name
                        #out_file = out_file.replace('.wav', '')

                        #for i, chunk in enumerate(chunks):
                        Denoise_2(file_name,out_file)

                except Exception as e:
                    print("[Error] Splitting error in %s. %s" % (file_name, e))
                    continue

if __name__ == '__main__': main()