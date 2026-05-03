import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def apply_stft(audio_signal, sample_rate):
    """Applies STFT to transform audio to frequency domain."""
    frequencies, times, Zxx = signal.stft(audio_signal, fs=sample_rate, nperseg=1024)
    return frequencies, times, Zxx

def split_into_bands(frequencies, Zxx):
    """Splits frequencies into Low, Mid, and High bands as required."""
    low_band_idx = frequencies < 500
    mid_band_idx = (frequencies >= 500) & (frequencies < 4000)
    high_band_idx = frequencies >= 4000

    return Zxx[low_band_idx, :], Zxx[mid_band_idx, :], Zxx[high_band_idx, :]

def plot_spectrogram(frequencies, times, Zxx):
    """Plots Spectrogram with a focus on 0-2000Hz for better clarity."""
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(times, frequencies, np.abs(Zxx), shading='gouraud')
    plt.title('STFT Spectrogram (Frequency vs Time)')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (seconds)')
    plt.colorbar(label='Magnitude')
    plt.ylim(0, 2000) # "The Trick" from your script for clarity
    plt.show()