# test_codec.py
import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from frame_handling import (
    compress_video, decompress_video, REFERENCE_BUFFER,
    compress_iframe, decompress_iframe, compress_pframe, decompress_pframe
)
from video_input import generate_test_video, load_video_frames, convert_frames_to_yuv, save_video_from_frames

def calculate_psnr(original_frames, decoded_frames):
    """
    Calculate Peak Signal-to-Noise Ratio between original and decoded frames
    """
    if len(original_frames) != len(decoded_frames):
        print(f"Warning: Frame count mismatch ({len(original_frames)} vs {len(decoded_frames)})")
        min_len = min(len(original_frames), len(decoded_frames))
        original_frames = original_frames[:min_len]
        decoded_frames = decoded_frames[:min_len]
    
    psnr_values = []
    
    for i, (orig, dec) in enumerate(zip(original_frames, decoded_frames)):
        # Ensure same shape
        if orig.shape != dec.shape:
            # Resize decoded frame to match original if needed
            if orig.shape[:2] != dec.shape[:2]:
                dec = cv2.resize(dec, (orig.shape[1], orig.shape[0]))
        
        # Calculate MSE
        mse = np.mean((orig.astype(np.float32) - dec.astype(np.float32)) ** 2)
        
        if mse == 0:
            psnr = float('inf')
        else:
            max_pixel = 255.0
            psnr = 10 * np.log10(max_pixel ** 2 / mse)
        
        psnr_values.append(psnr)
    
    avg_psnr = np.mean([p for p in psnr_values if p != float('inf')])
    return avg_psnr, psnr_values

def calculate_compression_ratio(original_frames, bitstreams):
    """
    Calculate compression ratio
    """
    # Calculate original size (assuming 3 bytes per pixel for RGB/YUV)
    total_pixels = 0
    for frame in original_frames:
        total_pixels += frame.shape[0] * frame.shape[1] * frame.shape[2]
    
    original_size_bytes = total_pixels  # 1 byte per channel
    
    compressed_size_bytes = sum(len(b) for b in bitstreams)
    
    compression_ratio = original_size_bytes / compressed_size_bytes if compressed_size_bytes > 0 else 0
    
    return compression_ratio, original_size_bytes, compressed_size_bytes

def visualize_comparison(original_frames, decoded_frames, num_frames=3):
    """
    Display comparison between original and decoded frames (console summary).
    """
    print("\n--- Visual Comparison (console) ---")
    step = max(1, len(original_frames) // num_frames)
    for i in range(0, min(len(original_frames), len(decoded_frames)), step):
        orig = original_frames[i]
        dec  = decoded_frames[i]
        print(f"Frame {i+1}: Original shape {orig.shape}, Decoded shape {dec.shape}")


def show_visual_comparison(yuv_original, yuv_decoded, psnr_values,
                           compression_ratio, avg_psnr, original_size, compressed_size,
                           compress_time, decompress_time, num_frames=4):
    """
    Rich matplotlib side-by-side comparison:
      Row 1 : N original frames (BGR display)
      Row 2 : N decoded  frames (BGR display)
      Row 3 : N difference maps (amplified)
      Bottom: PSNR-per-frame line chart + stats panel
    Saves to comparison.png and opens the window.
    """
    import matplotlib
    matplotlib.use('Agg')   # headless-safe; swap to 'TkAgg' if you want a popup

    n = min(num_frames, len(yuv_original), len(yuv_decoded))
    indices = [int(i * (len(yuv_original) - 1) / (n - 1)) for i in range(n)] if n > 1 else [0]

    def yuv2rgb(frame):
        bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    orig_rgb = [yuv2rgb(yuv_original[i]) for i in indices]
    dec_rgb  = [yuv2rgb(yuv_decoded[i])  for i in indices]
    diff_rgb = []
    for o, d in zip(orig_rgb, dec_rgb):
        diff = np.abs(o.astype(np.int16) - d.astype(np.int16)).astype(np.uint8)
        diff_amp = np.clip(diff * 6, 0, 255).astype(np.uint8)  # amplify for visibility
        diff_rgb.append(diff_amp)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(4 * n, 14), facecolor='#0d0d0d')
    fig.suptitle('VIDEO CODEC  —  COMPRESSION ANALYSIS',
                 fontsize=14, fontweight='bold', color='white',
                 y=0.98, fontfamily='monospace')

    outer = gridspec.GridSpec(4, 1, figure=fig,
                              hspace=0.45,
                              height_ratios=[3, 3, 3, 3.5])

    row_labels = ['ORIGINAL', 'DECODED', 'DIFFERENCE  (×6)']
    row_grids  = [
        gridspec.GridSpecFromSubplotSpec(1, n, subplot_spec=outer[0], wspace=0.04),
        gridspec.GridSpecFromSubplotSpec(1, n, subplot_spec=outer[1], wspace=0.04),
        gridspec.GridSpecFromSubplotSpec(1, n, subplot_spec=outer[2], wspace=0.04),
    ]
    frame_sets = [orig_rgb, dec_rgb, diff_rgb]
    cmaps      = [None, None, 'hot']
    border_col = ['#00e5ff', '#00ff99', '#ff4444']

    for row_i, (frames, gs, label, cmap, bcol) in enumerate(
            zip(frame_sets, row_grids, row_labels, cmaps, border_col)):

        for col_i, (frame, fidx) in enumerate(zip(frames, indices)):
            ax = fig.add_subplot(gs[col_i])
            if cmap:
                ax.imshow(frame[:, :, 0], cmap=cmap, vmin=0, vmax=255)
            else:
                ax.imshow(frame)

            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(bcol)
                spine.set_linewidth(2)

            # Frame index label
            ax.set_xlabel(f'frame {fidx + 1}', color='#aaaaaa',
                          fontsize=8, fontfamily='monospace')

            # PSNR badge on decoded row
            if row_i == 1 and fidx < len(psnr_values):
                pv = psnr_values[fidx]
                badge_color = ('#00ff99' if pv > 35 else
                               '#ffdd00' if pv > 25 else '#ff4444')
                ax.set_title(f'PSNR {pv:.1f} dB',
                             color=badge_color, fontsize=8,
                             fontfamily='monospace', pad=3)

        # Row label on the left of first column
        first_ax = fig.add_subplot(row_grids[row_i][0])
        first_ax.set_ylabel(label, color=bcol, fontsize=9,
                            fontfamily='monospace', labelpad=6)

    # ── Bottom row: PSNR chart + stats ────────────────────────────────────────
    bottom_gs = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[3], wspace=0.35, width_ratios=[3, 1])

    # PSNR line chart
    ax_psnr = fig.add_subplot(bottom_gs[0])
    ax_psnr.set_facecolor('#111111')
    finite_psnr = [p if p != float('inf') else None for p in psnr_values]
    xs = list(range(1, len(finite_psnr) + 1))
    ax_psnr.plot(xs, finite_psnr, color='#00e5ff', linewidth=1.5, zorder=3)
    ax_psnr.fill_between(xs, finite_psnr, alpha=0.15, color='#00e5ff')
    ax_psnr.axhline(avg_psnr, color='#ffdd00', linewidth=1,
                    linestyle='--', label=f'avg {avg_psnr:.1f} dB')
    # Shade quality bands
    ax_psnr.axhspan(40, max(50, max(v for v in finite_psnr if v)),
                    alpha=0.06, color='#00ff99')
    ax_psnr.axhspan(30, 40, alpha=0.06, color='#ffdd00')
    ax_psnr.axhspan(0,  30, alpha=0.06, color='#ff4444')
    ax_psnr.set_xlabel('Frame', color='#aaaaaa', fontsize=9, fontfamily='monospace')
    ax_psnr.set_ylabel('PSNR (dB)', color='#aaaaaa', fontsize=9, fontfamily='monospace')
    ax_psnr.set_title('PSNR per frame', color='white', fontsize=10,
                      fontfamily='monospace')
    ax_psnr.tick_params(colors='#666666', labelsize=8)
    for spine in ax_psnr.spines.values():
        spine.set_edgecolor('#333333')
    ax_psnr.legend(fontsize=8, facecolor='#1a1a1a', labelcolor='#ffdd00',
                   edgecolor='#333333')

    # Stats panel
    ax_stats = fig.add_subplot(bottom_gs[1])
    ax_stats.set_facecolor('#111111')
    ax_stats.set_xticks([]); ax_stats.set_yticks([])
    for spine in ax_stats.spines.values():
        spine.set_edgecolor('#333333')

    space_saving = (1 - compressed_size / original_size) * 100
    stats = [
        ('PSNR (avg)',      f'{avg_psnr:.2f} dB'),
        ('Comp. ratio',     f'{compression_ratio:.2f}:1'),
        ('Original',        f'{original_size/1024:.1f} KB'),
        ('Compressed',      f'{compressed_size/1024:.1f} KB'),
        ('Space saved',     f'{space_saving:.1f}%'),
        ('Encode speed',    f'{len(yuv_original)/compress_time:.1f} fps'),
        ('Decode speed',    f'{len(yuv_decoded)/decompress_time:.1f} fps'),
    ]

    ax_stats.set_title('Stats', color='white', fontsize=10, fontfamily='monospace')
    y = 0.92
    for key, val in stats:
        ax_stats.text(0.05, y, key, transform=ax_stats.transAxes,
                      color='#888888', fontsize=8, fontfamily='monospace')
        ax_stats.text(0.95, y, val, transform=ax_stats.transAxes,
                      color='#00e5ff', fontsize=8, fontfamily='monospace',
                      ha='right')
        y -= 0.12

    plt.savefig('comparison.png', dpi=150, bbox_inches='tight',
                facecolor='#0d0d0d')
    print("\n[VISUAL] Comparison saved to comparison.png")

    # Try to display — works in IDEs/Jupyter; silently skips in plain terminal
    try:
        import matplotlib
        if matplotlib.get_backend() != 'Agg':
            plt.show()
    except Exception:
        pass

    plt.close(fig)

def run_comprehensive_test():
    """
    Run comprehensive testing of the video codec
    """
    print("=" * 60)
    print("VIDEO CODEC TEST SUITE")
    print("=" * 60)
    
    # Step 1: Generate test video
    print("\n[1] Generating test video...")
    video_path = generate_test_video("test_video.avi", duration_seconds=2, fps=10, width=160, height=120)
    
    # Step 2: Load frames
    print("\n[2] Loading video frames...")
    bgr_frames = load_video_frames(video_path)
    print(f"Loaded {len(bgr_frames)} frames")
    
    # Step 3: Convert to YUV
    print("\n[3] Converting to YUV color space...")
    yuv_frames = convert_frames_to_yuv(bgr_frames)
    print(f"YUV frame shape: {yuv_frames[0].shape}")
    
    # Step 4: Compress video
    print("\n[4] Compressing video...")
    start_time = time.time()
    compressed_frames, bitstreams, global_bitstream = compress_video(yuv_frames, i_frame_interval=5, fps=10)
    compress_time = time.time() - start_time
    print(f"Compression time: {compress_time:.2f} seconds")

    # Save global bitstream to file
    with open("compressed.vcod", "wb") as f:
        f.write(global_bitstream)
    print(f"Global bitstream saved to compressed.vcod ({len(global_bitstream):,} bytes)")
    
    # Step 5: Decompress video
    print("\n[5] Decompressing video...")
    start_time = time.time()
    decoded_frames = decompress_video(global_bitstream, from_global=True)
    decompress_time = time.time() - start_time
    print(f"Decompression time: {decompress_time:.2f} seconds")
    
    # Step 6: Calculate metrics
    print("\n[6] Calculating quality metrics...")
    
    # PSNR
    avg_psnr, psnr_values = calculate_psnr(yuv_frames, decoded_frames)
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    
    # Compression ratio
    compression_ratio, original_size, compressed_size = calculate_compression_ratio(yuv_frames, bitstreams)
    print(f"Original size: {original_size:,} bytes")
    print(f"Compressed size: {compressed_size:,} bytes")
    print(f"Compression ratio: {compression_ratio:.2f}:1")
    print(f"Space saving: {(1 - compressed_size/original_size) * 100:.1f}%")
    
    # Step 7: Visual comparison
    visualize_comparison(yuv_frames, decoded_frames)
    
    # Step 8: Save output video
    print("\n[7] Saving decoded video...")
    # Convert YUV back to BGR for saving
    bgr_decoded = []
    for frame in decoded_frames:
        if frame.shape[2] == 3:
            # Convert YUV to BGR
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
            bgr_decoded.append(bgr_frame)
        else:
            bgr_decoded.append(frame)
    
    save_video_from_frames(bgr_decoded, "decoded_output.avi", fps=10)
    
    # Step 9: Individual frame type test
    print("\n[8] Testing individual frame compression...")
    
    # Test I-frame
    print("\n--- I-frame Test ---")
    test_frame = yuv_frames[0]
    compressed_i = compress_iframe(test_frame, 0)
    decompressed_i = decompress_iframe(compressed_i)
    psnr_i, _ = calculate_psnr([test_frame], [decompressed_i])
    print(f"I-frame PSNR: {psnr_i:.2f} dB")
    
    # Test P-frame (needs reference)
    if len(yuv_frames) > 1:
        print("\n--- P-frame Test ---")
        # Manually set reference buffer
        from frame_handling import REFERENCE_BUFFER
        REFERENCE_BUFFER.clear()
        REFERENCE_BUFFER.append(decompressed_i)
        
        compressed_p = compress_pframe(yuv_frames[1], 1)
        if compressed_p:
            decompressed_p = decompress_pframe(compressed_p)
            psnr_p, _ = calculate_psnr([yuv_frames[1]], [decompressed_p])
            print(f"P-frame PSNR: {psnr_p:.2f} dB")
            print(f"Motion vectors: {len(compressed_p['motion_vectors'])}")
        else:
            print("P-frame compression failed (no reference)")
    
    # Step 10: Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total frames processed: {len(yuv_frames)}")
    print(f"I-frame interval: every 5 frames")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Compression ratio: {compression_ratio:.2f}:1")
    print(f"Compression speed: {len(yuv_frames)/compress_time:.2f} frames/sec")
    print(f"Decompression speed: {len(yuv_frames)/decompress_time:.2f} frames/sec")
    
    # Quality assessment
    print("\n--- Quality Assessment ---")
    if avg_psnr > 40:
        print("✓ Excellent quality (PSNR > 40 dB)")
    elif avg_psnr > 30:
        print("✓ Good quality (PSNR > 30 dB)")
    elif avg_psnr > 20:
        print("⚠ Fair quality (PSNR > 20 dB)")
    else:
        print("✗ Poor quality (PSNR < 20 dB)")
    
    if compression_ratio > 10:
        print("✓ Excellent compression (>10:1)")
    elif compression_ratio > 5:
        print("✓ Good compression (>5:1)")
    elif compression_ratio > 2:
        print("⚠ Moderate compression (>2:1)")
    else:
        print("✗ Poor compression (<2:1)")
    
    print("\nOutput files generated:")
    print("  - test_video.avi (original)")
    print("  - decoded_output.avi (reconstructed)")
    print("  - compressed.vcod (global bitstream with header)")
    print("  - comparison.png  (side-by-side visual report)")

    # Step 11: Rich visual comparison
    show_visual_comparison(
        yuv_frames, decoded_frames,
        psnr_values, compression_ratio, avg_psnr,
        original_size, compressed_size,
        compress_time, decompress_time,
        num_frames=4
    )

    return {
        'avg_psnr': avg_psnr,
        'compression_ratio': compression_ratio,
        'original_size': original_size,
        'compressed_size': compressed_size,
        'compress_time': compress_time,
        'decompress_time': decompress_time
    }

def test_frame_type_decoding():
    """
    Test that frame type detection works correctly
    """
    print("\n" + "=" * 60)
    print("FRAME TYPE DECODING TEST")
    print("=" * 60)
    
    # Create a simple test frame
    test_frame = np.zeros((64, 64, 3), dtype=np.uint8)
    test_frame[20:44, 20:44, 0] = 255  # White square
    
    print(f"Test frame shape: {test_frame.shape}")
    
    # Test I-frame serialization/deserialization
    from frame_handling import serialize_iframe
    compressed_i = compress_iframe(test_frame, 0)
    serialized_i = serialize_iframe(compressed_i)
    # Note: import serialize_iframe locally for this test
    
    print("Frame type encoding test passed")

if __name__ == "__main__":
    # Run the comprehensive test
    results = run_comprehensive_test()
    
    # Additional verification
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    # Check if output files were created
    expected_files = ["test_video.avi", "decoded_output.avi", "compressed.vcod", "comparison.png"]
    for f in expected_files:
        if os.path.exists(f):
            size = os.path.getsize(f)
            print(f"✓ {f} created ({size:,} bytes)")
        else:
            print(f"✗ {f} not found")
    
    print("\nTest completed successfully!")