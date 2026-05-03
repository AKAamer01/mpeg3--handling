import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def generate_audio_data(duration=3, sample_rate=44100):
    """Generates audio: Pure Tone, Silence, and Noisy Signal."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # 1. Pure Signal (440Hz Sine Wave)
    pure_signal = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # 2. White Noise
    noise = 0.2 * np.random.normal(0, 0.5, pure_signal.shape)
    
    # 3. Create combined signal with noise and silence
    # Start with pure signal, add noise at the end, and silence in the middle
    combined_signal = np.copy(pure_signal) + noise # Add noise throughout
    
    # Silence from 1s to 2s
    silence_start = 1 * sample_rate
    silence_end = 2 * sample_rate
    combined_signal[silence_start:silence_end] = 0 
    
    return t, pure_signal, combined_signal, sample_rate

def save_wav_file(filename, signal, sample_rate):
    """Saves as 16-bit PCM WAV."""
    signal_16bit = np.int16(signal / np.max(np.abs(signal)) * 32767)
    wavfile.write(filename, sample_rate, signal_16bit)

def plot_signals(t, pure_signal, combined_signal):
    """Plots original, noisy, and signal with silence (3 subplots)."""
    plt.figure(figsize=(12, 8))

    # Plot 1: Pure Signal
    plt.subplot(3, 1, 1)
    plt.plot(t, pure_signal, color='blue')
    plt.title('Pure Signal (Original)')

    # Plot 2: Noisy Signal (Combined before silence)
    plt.subplot(3, 1, 2)
    plt.plot(t, combined_signal + 0.1, color='orange') # Showing logic
    plt.title('Noisy Signal')

    # Plot 3: Final Signal with Silence
    plt.subplot(3, 1, 3)
    plt.plot(t, combined_signal, color='red')
    plt.title('Final Signal (Noise + Silence 1s-2s)')
    plt.xlabel('Time (seconds)')

    plt.tight_layout()
    plt.show() # This will pop up the window