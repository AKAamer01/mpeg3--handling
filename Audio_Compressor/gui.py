import tkinter as tk
from tkinter import scrolledtext, ttk
import winsound # To play audio directly
from step1_handling import *
from step2_transform import *
from step3_quantization import *
from step4_encoding import *
from step5_evaluation import *

class AudioCompressorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Compressor Pro")
        self.root.geometry("700x600")
        
        tk.Label(root, text="MP3-Like Audio Compressor", font=("Arial", 16, "bold")).pack(pady=10)
        
        self.log_area = scrolledtext.ScrolledText(root, width=80, height=15, bg="#1e1e1e", fg="#00ff00")
        self.log_area.pack(pady=10)
        
        self.run_btn = ttk.Button(root, text="Start Compression Pipeline", command=self.run_pipeline)
        self.run_btn.pack(pady=5)

        # New Buttons for Sound
        self.play_orig_btn = ttk.Button(root, text="Play Original WAV", command=lambda: winsound.PlaySound("test_audio_original.wav", winsound.SND_FILENAME), state=tk.DISABLED)
        self.play_orig_btn.pack(pady=2)
        
        self.play_comp_btn = ttk.Button(root, text="Play Decompressed WAV", command=lambda: winsound.PlaySound("test_audio_decompressed.wav", winsound.SND_FILENAME), state=tk.DISABLED)
        self.play_comp_btn.pack(pady=2)

    def log(self, msg):
        self.log_area.insert(tk.END, msg + "\n")
        self.log_area.see(tk.END)
        self.root.update()

    def run_pipeline(self):
        self.log("🚀 Starting...")
        t, pure, noisy, fs = generate_audio_data()
        save_wav_file("test_audio_original.wav", noisy, fs)
        
        # This will pop up the 3-subplot window
        self.log("[Step 1] Plotting time-domain signals...")
        plot_signals(t, pure, noisy) 

        freqs, times, Zxx = apply_stft(noisy, fs)
        low, mid, high = split_into_bands(freqs, Zxx)
        
        # This will pop up the Spectrogram window
        self.log("[Step 2] Plotting Spectrogram (0-2000Hz focus)...")
        plot_spectrogram(freqs, times, Zxx)

        self.log("[Step 3] Quantizing (Bit Allocation)...")
        (l_q, l_max), (m_q, m_max), (h_q, h_max) = bit_allocation_and_quantize(low, mid, high)

        self.log("[Step 4] RLE Encoding...")
        l_enc = rle_encode(l_q)
        # (Add logic to show compression % here)
        
        self.log("[Step 5] Reconstructing...")
        rec = reconstruct_audio(dequantize_band(rle_decode(l_enc, l_q.shape), l_max, 8), mid, high, fs)
        save_wav_file("test_audio_decompressed.wav", rec, fs)
        
        snr = calculate_snr(noisy, rec)
        self.log(f"✅ Finished! Final SNR: {snr:.2f} dB")
        
        # Enable Sound Buttons
        self.play_orig_btn.config(state=tk.NORMAL)
        self.play_comp_btn.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioCompressorGUI(root)
    root.mainloop()