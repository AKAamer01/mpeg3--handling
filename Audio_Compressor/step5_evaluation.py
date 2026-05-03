import numpy as np
from scipy import signal

def calculate_snr(orig, decomp):
    min_l = min(len(orig), len(decomp))
    s_p = np.sum(orig[:min_l]**2)
    n_p = np.sum((orig[:min_l] - decomp[:min_l])**2)
    return 10 * np.log10(s_p / n_p) if n_p > 0 else float('inf')

def reconstruct_audio(l, m, h, fs):
    _, rec = signal.istft(np.vstack((l, m, h)), fs=fs, nperseg=1024)
    return rec