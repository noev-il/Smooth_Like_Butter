### Test Code for Plotting Spectogram over Time
import pdb

import librosa
from librosa.feature import chroma_stft
import librosa.display

import numpy as np
import pandas as pd

from scipy.stats import ttest_ind

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

import seaborn as sns

y, sr = librosa.load("D:\Spotify_API_Project\Audio_Samples\Bruno_Mars_Silk_Sonic_Intro.mp3", sr=41000)
tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='time', trim=False)
fig = plt.figure(figsize=(15, 8))
librosa.display.waveshow(y=y, sr=sr)
plt.vlines(beats, -1, 1, color='r',linestyles="null")
plt.grid()
#### plotting
"""This can be ignored. Old code from Audio Analyzer
df = pd.DataFrame(np.abs(librosa.stft(y, n_fft=256)))
bins = librosa.fft_frequencies(n_fft=256)

df['bins'] = bins / 1000
df['average_amplitude'] = df.mean(axis=1)
df = df[['bins', 'average_amplitude']]
max_freq = 20000
window = df.loc[(df.bins * 1000. >= 0) & (df.bins * 1000. <= max_freq)].copy()
window['scaled_amplitude'] = np.interp(window.average_amplitude, (0., max(window.average_amplitude)), (0., 1.))
window.plot(x='bins', y='scaled_amplitude', figsize=(16,4))
legend = plt.legend()
legend.remove()
plt.xlabel("Frequency (kHz)", fontsize=20)
plt.ylabel("Amplitude (scaled)", fontsize=20)
plt.title("Name", fontsize=26) """