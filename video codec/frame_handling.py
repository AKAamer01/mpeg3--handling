import cv2
import numpy as np

#todo: - implement bitstream handling for compressed data

# Constants and buffer
QUANT_MATRIX = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.int32)

ZIGZAG_ORDER = np.array([
    0, 1, 8, 16, 9, 2, 3, 10,
    17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
], dtype=np.int32)

ZIGZAG_INVERSE = np.argsort(ZIGZAG_ORDER)

REFERENCE_BUFFER = []

# main decision function
def frame_decision(frames):
    
    for index, frame in enumerate(frames, start=1):
        if (index - 1) % 6 == 0:
            compressed = handle_iframe(frame, index)
            REFERENCE_BUFFER.clear()
            decoded = iframe_decode(compressed)
            REFERENCE_BUFFER.append(decoded)
        else:
            compressed = handle_pframe(frame, index)
            decoded = pframe_decode(compressed)
            REFERENCE_BUFFER.append(decoded)

# handling functions
def handle_iframe(frame, index):
    # Extract Y channel for compression
    Y = frame[:, :, 0]
    height, width = Y.shape
    
    compressed_blocks = []
    
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = Y[i:i+8, j:j+8]
            # Apply DCT
            dct_block = cv2.dct(block.astype(np.float32))
            # Quantize
            quant_block = np.round(dct_block / QUANT_MATRIX).astype(np.int32)
            # Zigzag scan
            zigzag = quant_block.flatten()[ZIGZAG_ORDER]
            # Run-length encoding
            rle = run_length_encode(zigzag)
            compressed_blocks.append(rle)
    
    compressed_frame = {
        'index': index,
        'shape': (height, width),
        'compressed_blocks': compressed_blocks
    }
    
    print(f"Compressed I-frame at index {index}")
    return compressed_frame

def handle_pframe(frame, index):
    
    reference = REFERENCE_BUFFER[-1]
    if len(REFERENCE_BUFFER) != 1:
        print(f"No reference frame available for P-frame at index {index}")
        return None
    Y_current = frame[:, :, 0]
    Y_ref = reference[:, :, 0]
    height, width = Y_current.shape
    
    motion_vectors = []
    residual_blocks = []
    
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            current_block = Y_current[i:i+8, j:j+8]
            dx, dy = find_motion_vector(current_block, Y_ref, i, j)
            motion_vectors.append((dx, dy))
            
            predicted_block = Y_ref[i+dy:i+dy+8, j+dx:j+dx+8]
            residual = current_block.astype(np.int32) - predicted_block.astype(np.int32)
            
            # Apply DCT to residual
            dct_res = cv2.dct(residual.astype(np.float32))
            # Quantize
            quant_res = np.round(dct_res / QUANT_MATRIX).astype(np.int32)
            # Zigzag scan
            zigzag_res = quant_res.flatten()[ZIGZAG_ORDER]
            # Run-length encoding
            rle_res = run_length_encode(zigzag_res)
            residual_blocks.append(rle_res)
    
    compressed_pframe = {
        'index': index,
        'motion_vectors': motion_vectors,
        'residuals': residual_blocks
    }
    
    print(f"Compressed P-frame at index {index}")
    return compressed_pframe

# decoding functions
def iframe_decode(compressed_iframe):
    height, width = compressed_iframe['shape']
    decoded_Y = np.zeros((height, width), dtype=np.uint8)
    blocks = compressed_iframe['compressed_blocks']

    for block_index, rle in enumerate(blocks):
        zigzag = run_length_decode(rle)
        if len(zigzag) != 64:
            raise ValueError(f"Expected 64 decoded coefficients, got {len(zigzag)}")

        quant_flat = np.zeros(64, dtype=np.int32)
        quant_flat[ZIGZAG_ORDER] = zigzag
        quant_block = quant_flat.reshape((8, 8))

        dequant_block = quant_block * QUANT_MATRIX
        block = cv2.idct(dequant_block.astype(np.float32))
        block = np.clip(np.round(block), 0, 255).astype(np.uint8)

        row = (block_index // (width // 8)) * 8
        col = (block_index % (width // 8)) * 8
        decoded_Y[row:row + 8, col:col + 8] = block

    decoded_frame = np.zeros((height, width, 3), dtype=np.uint8)
    decoded_frame[:, :, 0] = decoded_Y
    decoded_frame[:, :, 1] = 128
    decoded_frame[:, :, 2] = 128

    return decoded_frame

def pframe_decode(compressed_pframe):
    
    reference = REFERENCE_BUFFER[-1]
    Y_ref = reference[:, :, 0]
    height, width = Y_ref.shape
    
    motion_vectors = compressed_pframe['motion_vectors']
    residuals = compressed_pframe['residuals']
    
    decoded_Y = np.zeros((height, width), dtype=np.uint8)
    
    for block_index, (dx, dy) in enumerate(motion_vectors):
        i = (block_index // (width // 8)) * 8
        j = (block_index % (width // 8)) * 8
        
        predicted_block = Y_ref[i + dy:i + dy + 8, j + dx:j + dx + 8]
        
        rle = residuals[block_index]
        zigzag = run_length_decode(rle)
        if len(zigzag) != 64:
            raise ValueError(f"Expected 64 decoded coefficients, got {len(zigzag)}")
        
        quant_flat = np.zeros(64, dtype=np.int32)
        quant_flat[ZIGZAG_ORDER] = zigzag
        quant_block = quant_flat.reshape((8, 8))
        
        dequant_block = quant_block * QUANT_MATRIX
        residual_block = cv2.idct(dequant_block.astype(np.float32))
        
        reconstructed_block = predicted_block.astype(np.float32) + residual_block
        reconstructed_block = np.clip(np.round(reconstructed_block), 0, 255).astype(np.uint8)
        
        decoded_Y[i:i + 8, j:j + 8] = reconstructed_block
    
    decoded_frame = np.zeros((height, width, 3), dtype=np.uint8)
    decoded_frame[:, :, 0] = decoded_Y
    decoded_frame[:, :, 1] = 128
    decoded_frame[:, :, 2] = 128
    
    return decoded_frame

# utility functions
def run_length_encode(arr):
    rle = []
    i = 0
    while i < len(arr):
        if arr[i] == 0:
            count = 0
            while i < len(arr) and arr[i] == 0:
                count += 1
                i += 1
            if count > 0:
                rle.append((0, count))
        else:
            rle.append((arr[i], 1))
            i += 1
    return rle


def run_length_decode(rle):
    values = []
    for value, count in rle:
        if value == 0:
            values.extend([0] * count)
        else:
            values.extend([value] * count)
    return np.array(values, dtype=np.int32)

def find_motion_vector(current_block, ref_frame, block_row, block_col, search_range=8):
    min_sad = float('inf')
    best_dx, best_dy = 0, 0
    h, w = ref_frame.shape
    
    for dy in range(-search_range, search_range + 1):
        for dx in range(-search_range, search_range + 1):
            r = block_row + dy
            c = block_col + dx
            if 0 <= r <= h - 8 and 0 <= c <= w - 8:
                ref_block = ref_frame[r:r+8, c:c+8]
                sad = np.sum(np.abs(current_block - ref_block))
                if sad < min_sad:
                    min_sad = sad
                    best_dx, best_dy = dx, dy
    
    return best_dx, best_dy

