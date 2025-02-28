import numpy as np
import librosa
#add noise
def add_noise(data,x):
    noise = np.random.randn(len(data))
    data_noise = data + x * noise
    return data_noise

#time-shifting
def shift(data,x):
    return np.roll(data, x)

#stretching slowing fasting
def stretch(data, rate):
    data = librosa.effects.time_stretch(data, rate=rate)
    return data