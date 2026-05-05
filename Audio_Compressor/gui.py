import tkinter as tk
from tkinter import ttk, scrolledtext
import winsound
import threading
from step1_handling import *
from step2_transform import *
from step3_quantization import *
from step4_encoding import *
from step5_evaluation import *

class AudioApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Audio Compressor")
        self.geometry("800x600")
        self.configure(bg="#1e1e1e")
        self.setup_ui()

    def setup_ui(self):
        tk.Label(self, text="Audio Compressor Pro", font=("Arial", 16, "bold"), bg="#1e1e1e", fg="#00ff00").pack(pady=10)
        
        btn_frame = tk.Frame(self, bg="#1e1e1e")
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="Start Pipeline", command=self.on_start).pack(side=tk.LEFT, padx=5)
        
        self.btn_orig = ttk.Button(btn_frame, text="Play Original", command=lambda: winsound.PlaySound("test_audio_original.wav", winsound.SND_FILENAME), state=tk.DISABLED)
        self.btn_orig.pack(side=tk.LEFT, padx=5)
        
        self.btn_comp = ttk.Button(btn_frame, text="Play Decompressed", command=lambda: winsound.PlaySound("test_audio_decompressed.wav", winsound.SND_FILENAME), state=tk.DISABLED)
        self.btn_comp.pack(side=tk.LEFT, padx=5)
        
        self.log_area = scrolledtext.ScrolledText(self, width=90, height=25, bg="#2d2d2d", fg="#00ff00", font=("Consolas", 10))
        self.log_area.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

    def log(self, msg):
        self.log_area.insert(tk.END, msg + "\n")
        self.log_area.see(tk.END)
        self.update()

    def on_start(self):
        threading.Thread(target=self._process_worker, daemon=True).start()

    def _process_worker(self):
        self.log("Starting Audio Pipeline...")
        t, pure, noisy, fs = generate_audio_data()
        save_wav_file("test_audio_original.wav", noisy, fs)
        
        self.log("Plotting time-domain signals...")
        plot_signals(t, pure, noisy) 

        freqs, times, Zxx = apply_stft(noisy, fs)
        low, mid, high = split_into_bands(freqs, Zxx)
        
        self.log("Plotting Spectrogram...")
        plot_spectrogram(freqs, times, Zxx)

        self.log("Quantizing...")
        (l_q, l_max), (m_q, m_max), (h_q, h_max) = bit_allocation_and_quantize(low, mid, high)

        self.log("RLE Encoding...")
        l_enc = rle_encode(l_q)
        
        self.log("Reconstructing...")
        rec = reconstruct_audio(dequantize_band(rle_decode(l_enc, l_q.shape), l_max, 8), mid, high, fs)
        save_wav_file("test_audio_decompressed.wav", rec, fs)
        
        snr = calculate_snr(noisy, rec)
        self.log(f"Finished! Final SNR: {snr:.2f} dB")
        
        self.btn_orig.config(state=tk.NORMAL)
        self.btn_comp.config(state=tk.NORMAL)

if __name__ == "__main__":
    app = AudioApp()
    app.mainloop()