import numpy as np

# Importing functions from our project steps
from step1_handling import generate_audio_data, save_wav_file
from step2_transform import apply_stft, split_into_bands
from step3_quantization import bit_allocation_and_quantize, dequantize_band
from step4_encoding import rle_encode, rle_decode
from step5_evaluation import calculate_snr, reconstruct_audio

def main():
    print("="*50)
    print("🚀 Starting Audio Compression Project (MP3 Encoder) 🚀")
    print("="*50)

    # --- Step 1: Input Handling ---
    print("\n[1] Handling Audio Input (Generating Noise & Silence)...")
    time_array, clean_audio, noisy_audio, fs = generate_audio_data()
    save_wav_file("test_audio_original.wav", noisy_audio, fs)
    print("    -> Original file saved as: test_audio_original.wav")

    # --- Step 2: Transform to Frequency Domain ---
    print("\n[2] Transforming to Frequency Domain (STFT)...")
    freqs, times, Zxx = apply_stft(noisy_audio, fs)
    low_band, mid_band, high_band = split_into_bands(freqs, Zxx)
    print(f"    -> Bands Created: Low({low_band.shape[0]}), Mid({mid_band.shape[0]}), High({high_band.shape[0]})")

    # --- Step 3: Quantization & Bit Allocation ---
    print("\n[3] Performing Quantization & Bit Allocation...")
    (low_q, low_max), (mid_q, mid_max), (high_q, high_max) = bit_allocation_and_quantize(low_band, mid_band, high_band)
    print("    -> Bit Allocation: Low=8, Mid=6, High=4 bits")

    # --- Step 4: Encoding ---
    print("\n[4] Encoding Data using Run-Length Encoding (RLE)...")
    low_encoded = rle_encode(low_q)
    mid_encoded = rle_encode(mid_q)
    high_encoded = rle_encode(high_q)
    
    original_size = low_q.size + mid_q.size + high_q.size
    compressed_size = len(low_encoded) + len(mid_encoded) + len(high_encoded)
    print(f"    -> Original Size: {original_size} items")
    print(f"    -> Compressed Size: {compressed_size} pairs")
    print(f"    -> Compression Ratio: {(1 - (compressed_size / original_size)) * 100:.2f}%")

    # --- Step 5: Reconstruction & Evaluation ---
    print("\n[5] Decoding & Reconstructing Audio (Decompression)...")
    low_decoded = rle_decode(low_encoded, low_q.shape)
    mid_decoded = rle_decode(mid_encoded, mid_q.shape)
    high_decoded = rle_decode(high_encoded, high_q.shape)
    
    low_dq = dequantize_band(low_decoded, low_max, bits=8)
    mid_dq = dequantize_band(mid_decoded, mid_max, bits=6)
    high_dq = dequantize_band(high_decoded, high_max, bits=4)
    
    reconstructed_audio = reconstruct_audio(low_dq, mid_dq, high_dq, fs)
    save_wav_file("test_audio_decompressed.wav", reconstructed_audio, fs)
    
    print("\n[>] Calculating Signal-to-Noise Ratio (SNR)...")
    snr_value = calculate_snr(noisy_audio, reconstructed_audio)
    print(f"✅ Final SNR Value: {snr_value:.2f} dB")
    
    print("\n" + "="*50)
    print("🎉 Audio Compression Project Completed Successfully! 🎉")
    print("="*50)

if __name__ == "__main__":
    main()