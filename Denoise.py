import numpy as np
import scipy as sp
from scipy.io.wavfile import read
from scipy.io.wavfile import write     # Imported libaries such as numpy, scipy(read, write), matplotlib.pyplot
from scipy import signal
import matplotlib.pyplot as plt
import soundfile as sf


def denoise(sound_file):
    array, frequency = sf.read(sound_file,dtype='float32')  # Reading the sound file.
    print(array.shape)
    print('sample length ', len(array))
    plt.plot(array)
    plt.title('Original Signal Spectrum')
    plt.xlabel('Frequency(Hz)')
    plt.ylabel('Amplitude')
    plt.show()

    FourierTransformation = sp.fft(array)  # Calculating the fourier transformation of the signal

    # In[14]:

    scale = sp.linspace(0, frequency, len(array))

    # In[15]:


    # In[16]:


    #FourierTransformation = sp.fft(array)

    b, a = signal.butter(5, 1000 / (frequency / 2), btype='highpass')  # ButterWorth filter 4350

    # In[20]:

    filteredSignal = signal.lfilter(b, a, array)

    plt.plot(filteredSignal)  # plotting the signal.
    plt.title('Highpass Filter')
    plt.xlabel('Frequency(Hz)')
    plt.ylabel('Amplitude')
    plt.show()
    # In[21]:

    c, d = signal.butter(5, 380 / (frequency / 2), btype='lowpass')  # ButterWorth low-filter
    newFilteredSignal = signal.lfilter(c, d, filteredSignal)  # Applying the filter to the signal
    print(newFilteredSignal.shape)
    newFilteredSignal = np.divide(newFilteredSignal[0], max(newFilteredSignal[0]))
    print(newFilteredSignal)
    plt.plot(newFilteredSignal)  # plotting the signal.
    plt.title('Lowpass Filter')
    plt.xlabel('Frequency(Hz)')
    plt.ylabel('Amplitude')
    plt.show()

    # In[22]:

    sf.write('New-hammering-lee2.wav', newFilteredSignal, frequency)  # Saving it to the file.


def main():
    denoise('hammering_lee2.wav')  # Reading the sound file.




if __name__ == '__main__': main()