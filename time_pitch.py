### Test Code for Plotting Spectogram over Time
import pdb

import librosa
from librosa.feature import chroma_stft
import librosa.display

import numpy as np
import pandas as pd

from scipy.stats import ttest_ind

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
matplotlib.use('TkAgg')
import seaborn as sns
### IMPORT LIBRARIES
y, sr = librosa.load("After_the_Storm.mp3", sr=41000)
##LOAD FILE

D = librosa.stft(y)  # STFT of y
### Short-Time Fourier Transform
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
###STFT magnitude to decibles

plt.figure()
librosa.display.specshow(S_db)
plt.colorbar()

#### plotting

times = librosa.times_like(S_db)

# Calculate the average amplitude
average_amplitude = np.mean(np.abs(D), axis=0)

# Plot the spectrogram with time on the x-axis
plt.figure(figsize=(12, 8))
librosa.display.specshow(S_db, x_axis='time', sr=sr, hop_length=512)

# Add colorbar for amplitude scale
plt.colorbar(format='%+2.0f dB')
plt.xlabel("Time (s)", fontsize=20)
plt.ylabel("Frequency (Hz)", fontsize=20)
plt.title("Spectrogram over Time", fontsize=26)
plt.show()
