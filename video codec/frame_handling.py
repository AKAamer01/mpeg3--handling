import cv2
import numpy as np
import struct
from collections import Counter
import heapq

# JPEG Quantization Matrix (standard)
QUANT_MATRIX = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32)

# Zig-zag scan order for 8x8 blocks
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

# Global reference buffer for motion compensation
REFERENCE_BUFFER = []

# ================ HUFFMAN CODING IMPLEMENTATION ================

class HuffmanNode:
    def __init__(self, value, freq):
        self.value = value
        self.freq = freq
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.freq < other.freq

class HuffmanCoding:
    def __init__(self):
        self.codes = {}
        self.reverse_codes = {}
    
    def build_tree(self, data):
        """Build Huffman tree from data"""
        if not data:
            return None
        
        freq_counter = Counter(data)
        heap = [HuffmanNode(value, freq) for value, freq in freq_counter.items()]
        heapq.heapify(heap)
        
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            merged = HuffmanNode(None, left.freq + right.freq)
            merged.left = left
            merged.right = right
            heapq.heappush(heap, merged)
        
        return heap[0]
    
    def generate_codes(self, node, code=""):
        """Generate Huffman codes from tree"""
        if node is None:
            return
        
        if node.value is not None:
            self.codes[node.value] = code
            self.reverse_codes[code] = node.value
            return
        
        self.generate_codes(node.left, code + "0")
        self.generate_codes(node.right, code + "1")
    
    def encode(self, data):
        """Encode data using Huffman coding"""
        if not data:
            return b'', None, 0
        
        tree = self.build_tree(data)
        self.codes.clear()
        self.reverse_codes.clear()
        self.generate_codes(tree)
        
        encoded_bits = ''.join(self.codes[value] for value in data)
        
        padding = 8 - (len(encoded_bits) % 8)
        if padding == 8:
            padding = 0
        encoded_bits += '0' * padding
        
        encoded_bytes = bytearray()
        for i in range(0, len(encoded_bits), 8):
            byte = encoded_bits[i:i+8]
            encoded_bytes.append(int(byte, 2))
        
        tree_bytes = self.serialize_tree(tree)
        
        return bytes(encoded_bytes), tree_bytes, padding
    
    def decode(self, encoded_bytes, tree_bytes, padding):
        """Decode Huffman encoded data"""
        if not encoded_bytes:
            return []
        
        tree = self.deserialize_tree(tree_bytes)
        self.reverse_codes.clear()
        self.generate_codes(tree)
        
        bits = ""
        for byte in encoded_bytes:
            bits += format(byte, '08b')
        if padding > 0:
            bits = bits[:-padding]
        
        decoded = []
        current_code = ""
        for bit in bits:
            current_code += bit
            if current_code in self.reverse_codes:
                decoded.append(self.reverse_codes[current_code])
                current_code = ""
        
        return decoded
    
    def serialize_tree(self, node):
        """Serialize Huffman tree to bytes"""
        if node is None:
            return b''
        
        if node.value is not None:
            return struct.pack('>Bi', 1, node.value)
        else:
            left_bytes = self.serialize_tree(node.left)
            right_bytes = self.serialize_tree(node.right)
            return struct.pack('>B', 0) + left_bytes + right_bytes
    
    def deserialize_tree(self, data):
        """Deserialize Huffman tree from bytes"""
        self.pos = 0
        return self._deserialize_tree(data)
    
    def _deserialize_tree(self, data):
        if self.pos >= len(data):
            return None
        
        flag = struct.unpack('>B', data[self.pos:self.pos+1])[0]
        self.pos += 1
        
        if flag == 1:
            value = struct.unpack('>i', data[self.pos:self.pos+4])[0]
            self.pos += 4
            return HuffmanNode(value, 0)
        else:
            node = HuffmanNode(None, 0)
            node.left = self._deserialize_tree(data)
            node.right = self._deserialize_tree(data)
            return node

huffman_coder = HuffmanCoding()

# ================ RUN-LENGTH ENCODING ================

def run_length_encode(arr):
    """Apply run-length encoding to array"""
    rle = []
    i = 0
    arr = arr.tolist() if isinstance(arr, np.ndarray) else arr
    while i < len(arr):
        if arr[i] == 0:
            count = 0
            while i < len(arr) and arr[i] == 0:
                count += 1
                i += 1
            rle.append((0, count))
        else:
            rle.append((int(arr[i]), 1))
            i += 1
    return rle

def run_length_decode(rle):
    """Decode run-length encoded data"""
    values = []
    for value, count in rle:
        if value == 0:
            values.extend([0] * count)
        else:
            values.extend([value] * count)
    return np.array(values, dtype=np.int32)

# ================ MOTION ESTIMATION ================

def find_motion_vector(current_block, ref_frame, block_row, block_col, search_range=16):
    """Find motion vector using block matching (minimum SAD)"""
    min_sad = float('inf')
    best_dx, best_dy = 0, 0
    h, w = ref_frame.shape
    
    # Ensure current_block is the right shape
    if current_block.shape != (8, 8):
        return 0, 0
    
    for dy in range(-search_range, search_range + 1):
        for dx in range(-search_range, search_range + 1):
            r = block_row + dy
            c = block_col + dx
            if 0 <= r <= h - 8 and 0 <= c <= w - 8:
                ref_block = ref_frame[r:r+8, c:c+8]
                # Ensure both blocks have same dtype
                current_block_float = current_block.astype(np.float32)
                ref_block_float = ref_block.astype(np.float32)
                sad = np.sum(np.abs(current_block_float - ref_block_float))
                if sad < min_sad:
                    min_sad = sad
                    best_dx, best_dy = dx, dy
    
    return best_dx, best_dy

# ================ I-FRAME COMPRESSION (INTRA-FRAME) ================

def _compress_channel(channel):
    """
    Compress a single 2-D channel (Y, U, or V) using DCT + quantization +
    zig-zag scan + RLE.  Returns a list of RLE blocks.
    """
    height, width = channel.shape
    blocks = []
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = channel[i:i+8, j:j+8].astype(np.float32)
            if block.shape != (8, 8):
                block = np.pad(block,
                               ((0, 8 - block.shape[0]), (0, 8 - block.shape[1])),
                               'constant')
            dct_block   = cv2.dct(block)
            quant_block = np.round(dct_block / QUANT_MATRIX).astype(np.int32)
            zigzag      = quant_block.flatten()[ZIGZAG_ORDER]
            blocks.append(run_length_encode(zigzag))
    return blocks


def _decompress_channel(blocks, height, width):
    """
    Decompress a list of RLE blocks back to a 2-D channel (uint8).
    """
    channel      = np.zeros((height, width), dtype=np.uint8)
    cols_per_row = (width + 7) // 8

    for block_idx, rle in enumerate(blocks):
        zigzag = run_length_decode(rle)
        if len(zigzag) < 64:
            zigzag = np.pad(zigzag, (0, 64 - len(zigzag)), 'constant')

        quant_flat = np.zeros(64, dtype=np.int32)
        quant_flat[ZIGZAG_INVERSE] = zigzag[:64]
        dequant = quant_flat.reshape((8, 8)) * QUANT_MATRIX
        block   = cv2.idct(dequant.astype(np.float32))
        block   = np.clip(np.round(block), 0, 255).astype(np.uint8)

        row = (block_idx // cols_per_row) * 8
        col = (block_idx %  cols_per_row) * 8
        if row + 8 <= height and col + 8 <= width:
            channel[row:row+8, col:col+8] = block

    return channel


def compress_iframe(frame, index):
    """
    I-frame compression using DCT, quantization, zig-zag scan, and RLE.
    All three YUV channels (Y, U, V) are compressed independently.
    """
    height, width = frame.shape[:2]
    return {
        'type':              'I',
        'index':             index,
        'shape':             (height, width),
        'compressed_blocks':   _compress_channel(frame[:, :, 0]),  # luma Y
        'compressed_blocks_u': _compress_channel(frame[:, :, 1]),  # chroma U
        'compressed_blocks_v': _compress_channel(frame[:, :, 2]),  # chroma V
    }


def decompress_iframe(compressed_frame):
    """Decompress I-frame — reconstructs full YUV (Y + U + V channels)."""
    height, width = compressed_frame['shape']
    decoded_Y = _decompress_channel(compressed_frame['compressed_blocks'],   height, width)
    decoded_U = _decompress_channel(compressed_frame['compressed_blocks_u'], height, width)
    decoded_V = _decompress_channel(compressed_frame['compressed_blocks_v'], height, width)
    return np.stack([decoded_Y, decoded_U, decoded_V], axis=2)

# ================ P-FRAME COMPRESSION (INTER-FRAME) ================

def _compress_pframe_channel(current_ch, ref_ch, motion_vectors=None):
    """
    Compress one channel of a P-frame.
    If motion_vectors is None (Y channel), compute them via block matching and
    return (motion_vectors, residual_blocks).
    If motion_vectors is provided (U/V channels), reuse them and only return
    residual_blocks.
    """
    height, width = current_ch.shape
    cols_per_row  = (width + 7) // 8
    cur_f  = current_ch.astype(np.float32)
    ref_f  = ref_ch.astype(np.float32)
    compute_mv    = motion_vectors is None
    if compute_mv:
        motion_vectors = []
    residual_blocks = []

    block_idx = 0
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            cur_block = cur_f[i:i+8, j:j+8]
            if cur_block.shape != (8, 8):
                cur_block = np.pad(cur_block,
                                   ((0, 8-cur_block.shape[0]), (0, 8-cur_block.shape[1])),
                                   'constant')

            if compute_mv:
                dx, dy = find_motion_vector(cur_block, ref_f, i, j)
                motion_vectors.append((dx, dy))
            else:
                dx, dy = motion_vectors[block_idx]

            pred_i = max(0, min(i + dy, height - 8))
            pred_j = max(0, min(j + dx, width  - 8))
            pred_block = ref_f[pred_i:pred_i+8, pred_j:pred_j+8]

            residual    = cur_block - pred_block
            dct_res     = cv2.dct(residual.astype(np.float32))
            quant_res   = np.round(dct_res / QUANT_MATRIX).astype(np.int32)
            zigzag_res  = quant_res.flatten()[ZIGZAG_ORDER]
            residual_blocks.append(run_length_encode(zigzag_res))
            block_idx += 1

    if compute_mv:
        return motion_vectors, residual_blocks
    return residual_blocks


def _decompress_pframe_channel(motion_vectors, residuals, ref_ch):
    """
    Reconstruct one channel of a P-frame from motion vectors + residuals.
    """
    height, width = ref_ch.shape
    cols_per_row  = (width + 7) // 8
    ref_f         = ref_ch.astype(np.float32)
    decoded       = np.zeros((height, width), dtype=np.uint8)

    for block_idx, (dx, dy) in enumerate(motion_vectors):
        i = (block_idx // cols_per_row) * 8
        j = (block_idx %  cols_per_row) * 8

        pred_i = max(0, min(i + dy, height - 8))
        pred_j = max(0, min(j + dx, width  - 8))
        pred_block = ref_f[pred_i:pred_i+8, pred_j:pred_j+8]

        if block_idx < len(residuals):
            zigzag = run_length_decode(residuals[block_idx])
            if len(zigzag) < 64:
                zigzag = np.pad(zigzag, (0, 64 - len(zigzag)), 'constant')
            qf = np.zeros(64, dtype=np.int32)
            qf[ZIGZAG_INVERSE] = zigzag[:64]
            res_block = cv2.idct((qf.reshape((8, 8)) * QUANT_MATRIX).astype(np.float32))
        else:
            res_block = np.zeros((8, 8), dtype=np.float32)

        rec = np.clip(np.round(pred_block + res_block), 0, 255).astype(np.uint8)
        if i + 8 <= height and j + 8 <= width:
            decoded[i:i+8, j:j+8] = rec

    return decoded


def compress_pframe(frame, index):
    """
    P-frame compression using motion estimation, motion vectors, and residual
    encoding.  Motion vectors are derived from the Y channel and reused for
    U and V (chroma) channels.
    """
    if len(REFERENCE_BUFFER) == 0:
        return None

    reference = REFERENCE_BUFFER[-1]

    # Y channel — compute motion vectors here
    motion_vectors, residuals_y = _compress_pframe_channel(
        frame[:, :, 0], reference[:, :, 0])

    # U and V — reuse the same motion vectors
    residuals_u = _compress_pframe_channel(
        frame[:, :, 1], reference[:, :, 1], motion_vectors)
    residuals_v = _compress_pframe_channel(
        frame[:, :, 2], reference[:, :, 2], motion_vectors)

    return {
        'type':         'P',
        'index':        index,
        'motion_vectors': motion_vectors,
        'residuals':    residuals_y,   # Y residuals
        'residuals_u':  residuals_u,   # U residuals
        'residuals_v':  residuals_v,   # V residuals
    }


def decompress_pframe(compressed_frame):
    """Decompress P-frame — reconstructs full YUV using motion compensation."""
    if len(REFERENCE_BUFFER) == 0:
        raise ValueError("No reference frame available")

    reference      = REFERENCE_BUFFER[-1]
    motion_vectors = compressed_frame['motion_vectors']

    decoded_Y = _decompress_pframe_channel(
        motion_vectors, compressed_frame['residuals'],   reference[:, :, 0])
    decoded_U = _decompress_pframe_channel(
        motion_vectors, compressed_frame.get('residuals_u', []), reference[:, :, 1])
    decoded_V = _decompress_pframe_channel(
        motion_vectors, compressed_frame.get('residuals_v', []), reference[:, :, 2])

    return np.stack([decoded_Y, decoded_U, decoded_V], axis=2)

# ================ BITSTREAM FORMATION ================

def _serialize_rle_blocks(blocks):
    """Huffman-encode a list of RLE blocks and return the packed bytes."""
    flat = []
    for block in blocks:
        for value, count in block:
            flat.append(value)
            flat.append(count)
    encoded, tree_bytes, padding = huffman_coder.encode(flat)
    buf = bytearray()
    buf.extend(struct.pack('>I', len(tree_bytes)))
    buf.extend(tree_bytes)
    buf.extend(struct.pack('>B', padding))
    buf.extend(struct.pack('>I', len(encoded)))
    buf.extend(encoded)
    return bytes(buf)


def _deserialize_rle_blocks(data, offset):
    """Decode one Huffman-encoded RLE block stream starting at offset.
    Returns (compressed_blocks, new_offset)."""
    tree_size = struct.unpack('>I', data[offset:offset+4])[0]; offset += 4
    tree_bytes = data[offset:offset+tree_size];                 offset += tree_size
    padding    = struct.unpack('>B', data[offset:offset+1])[0]; offset += 1
    data_size  = struct.unpack('>I', data[offset:offset+4])[0]; offset += 4
    encoded    = data[offset:offset+data_size];                 offset += data_size

    decoded_values = huffman_coder.decode(encoded, tree_bytes, padding)

    blocks = []
    idx = 0
    while idx + 1 < len(decoded_values):
        block = []
        total = 0
        while idx + 1 < len(decoded_values):
            v = decoded_values[idx]; c = decoded_values[idx+1]; idx += 2
            block.append((v, c))
            total += c if v == 0 else 1
            if total >= 64:
                break
        if block:
            blocks.append(block)
    return blocks, offset


def serialize_iframe(compressed_frame):
    """Serialize I-frame to bitstream with Huffman coding (Y + U + V channels)."""
    data = bytearray()

    # Frame header: type (1 byte), index (4 bytes), height (2 bytes), width (2 bytes)
    data.append(0x49)  # 'I'
    data.extend(struct.pack('>I', compressed_frame['index']))
    data.extend(struct.pack('>H', compressed_frame['shape'][0]))
    data.extend(struct.pack('>H', compressed_frame['shape'][1]))

    # Encode all three channels
    data.extend(_serialize_rle_blocks(compressed_frame['compressed_blocks']))    # Y
    data.extend(_serialize_rle_blocks(compressed_frame['compressed_blocks_u']))  # U
    data.extend(_serialize_rle_blocks(compressed_frame['compressed_blocks_v']))  # V

    return bytes(data)

def serialize_pframe(compressed_frame):
    """Serialize P-frame to bitstream with Huffman coding (Y + U + V residuals)."""
    data = bytearray()

    # Frame header: type (1 byte), index (4 bytes)
    data.append(0x50)  # 'P'
    data.extend(struct.pack('>I', compressed_frame['index']))

    # Motion vectors (shared across all channels)
    num_vectors = len(compressed_frame['motion_vectors'])
    data.extend(struct.pack('>I', num_vectors))

    flat_mv = []
    for dx, dy in compressed_frame['motion_vectors']:
        flat_mv.append(dx); flat_mv.append(dy)
    encoded_mv, mv_tree, mv_padding = huffman_coder.encode(flat_mv)
    data.extend(struct.pack('>I', len(mv_tree))); data.extend(mv_tree)
    data.extend(struct.pack('>B', mv_padding))
    data.extend(struct.pack('>I', len(encoded_mv))); data.extend(encoded_mv)

    # Residuals for Y, U, V channels
    data.extend(struct.pack('>I', len(compressed_frame['residuals'])))
    data.extend(_serialize_rle_blocks(compressed_frame['residuals']))   # Y
    data.extend(_serialize_rle_blocks(compressed_frame['residuals_u'])) # U
    data.extend(_serialize_rle_blocks(compressed_frame['residuals_v'])) # V

    return bytes(data)

def deserialize_frame(bitstream):
    """Deserialize a single frame from its bitstream bytes."""
    offset     = 0
    frame_type = bitstream[offset]; offset += 1

    if frame_type == 0x49:
        # ── I-frame ──────────────────────────────────────────────────────────
        index  = struct.unpack('>I', bitstream[offset:offset+4])[0]; offset += 4
        height = struct.unpack('>H', bitstream[offset:offset+2])[0]; offset += 2
        width  = struct.unpack('>H', bitstream[offset:offset+2])[0]; offset += 2

        blocks_y, offset = _deserialize_rle_blocks(bitstream, offset)
        blocks_u, offset = _deserialize_rle_blocks(bitstream, offset)
        blocks_v, offset = _deserialize_rle_blocks(bitstream, offset)

        return {
            'type':              'I',
            'index':             index,
            'shape':             (height, width),
            'compressed_blocks':   blocks_y,
            'compressed_blocks_u': blocks_u,
            'compressed_blocks_v': blocks_v,
        }

    else:
        # ── P-frame ──────────────────────────────────────────────────────────
        index       = struct.unpack('>I', bitstream[offset:offset+4])[0]; offset += 4
        num_vectors = struct.unpack('>I', bitstream[offset:offset+4])[0]; offset += 4

        # Motion vectors
        mv_tree_size = struct.unpack('>I', bitstream[offset:offset+4])[0]; offset += 4
        mv_tree      = bitstream[offset:offset+mv_tree_size];              offset += mv_tree_size
        mv_padding   = struct.unpack('>B', bitstream[offset:offset+1])[0]; offset += 1
        mv_data_size = struct.unpack('>I', bitstream[offset:offset+4])[0]; offset += 4
        mv_encoded   = bitstream[offset:offset+mv_data_size];              offset += mv_data_size

        mv_values      = huffman_coder.decode(mv_encoded, mv_tree, mv_padding)
        motion_vectors = [(mv_values[i], mv_values[i+1])
                          for i in range(0, len(mv_values), 2)][:num_vectors]

        # Residuals: Y, U, V
        num_residuals   = struct.unpack('>I', bitstream[offset:offset+4])[0]; offset += 4
        residuals_y, offset = _deserialize_rle_blocks(bitstream, offset)
        residuals_u, offset = _deserialize_rle_blocks(bitstream, offset)
        residuals_v, offset = _deserialize_rle_blocks(bitstream, offset)

        return {
            'type':           'P',
            'index':          index,
            'motion_vectors': motion_vectors,
            'residuals':      residuals_y[:num_residuals],
            'residuals_u':    residuals_u,
            'residuals_v':    residuals_v,
        }

# ================ MAIN VIDEO COMPRESSION PIPELINE ================

# Global bitstream magic number
_MAGIC = b'VCOD'   # 4-byte magic
_VERSION = 1


def build_bitstream(bitstreams, num_frames, fps, width, height, i_frame_interval):
    """
    Package per-frame bitstreams into a single binary bitstream with a
    global header.

    Header layout (22 bytes):
      4B  magic           'VCOD'
      1B  version         1
      2B  num_frames      total frame count
      1B  fps             frames per second
      2B  width           frame width  (pixels)
      2B  height          frame height (pixels)
      1B  i_interval      I-frame interval
      1B  num_frame_types len of frame-type index
      N*2B frame_type_index  (frame_index 2B, type 1B) — stored as
            array of (offset 4B, type 1B) pairs for each frame
      ... per-frame bitstreams concatenated

    For simplicity we store:
      4B  magic
      1B  version
      2B  num_frames
      1B  fps
      2B  width
      2B  height
      1B  i_interval
      Then for each frame: 1B frame_type_indicator ('I'=0x49 / 'P'=0x50), 4B frame_size
      Then frame data
    """
    buf = bytearray()
    # Global header
    buf.extend(_MAGIC)
    buf.append(_VERSION)
    buf.extend(struct.pack('>H', num_frames))
    buf.append(fps)
    buf.extend(struct.pack('>H', width))
    buf.extend(struct.pack('>H', height))
    buf.append(i_frame_interval)

    # Frame index: type byte + size for each frame
    for bs in bitstreams:
        frame_type_byte = bs[0]  # 0x49 or 0x50
        buf.append(frame_type_byte)
        buf.extend(struct.pack('>I', len(bs)))
        buf.extend(bs)

    return bytes(buf)


def parse_bitstream(data):
    """
    Parse a global bitstream back into per-frame bitstreams + metadata.
    Returns (bitstreams, metadata_dict).
    """
    offset = 0
    magic = data[offset:offset+4]; offset += 4
    if magic != _MAGIC:
        raise ValueError(f"Invalid magic: {magic!r} (expected {_MAGIC!r})")

    version      = data[offset]; offset += 1
    num_frames   = struct.unpack('>H', data[offset:offset+2])[0]; offset += 2
    fps          = data[offset]; offset += 1
    width        = struct.unpack('>H', data[offset:offset+2])[0]; offset += 2
    height       = struct.unpack('>H', data[offset:offset+2])[0]; offset += 2
    i_interval   = data[offset]; offset += 1

    metadata = {
        'version': version, 'num_frames': num_frames, 'fps': fps,
        'width': width, 'height': height, 'i_frame_interval': i_interval,
    }

    bitstreams   = []
    frame_types  = []
    for _ in range(num_frames):
        ftype = data[offset]; offset += 1
        fsize = struct.unpack('>I', data[offset:offset+4])[0]; offset += 4
        bitstreams.append(data[offset:offset+fsize]); offset += fsize
        frame_types.append(chr(ftype))

    metadata['frame_types'] = frame_types
    return bitstreams, metadata


def compress_video(frames, i_frame_interval=5, fps=10):
    """
    Main video compression pipeline.
    Returns (compressed_frames, bitstreams, global_bitstream).
    global_bitstream is a single bytes object with a complete header +
    per-frame data — ready to write to a .vcod file.
    """
    global REFERENCE_BUFFER
    REFERENCE_BUFFER = []

    compressed_frames = []
    bitstreams        = []

    height, width = frames[0].shape[:2]

    print(f"\nCompressing {len(frames)} frames...")
    print(f"Resolution : {width}x{height}  |  FPS: {fps}  |  I-interval: {i_frame_interval}")
    print(f"{'Frame':<8} {'Type':<14} {'Size (bytes)':>12}  {'Info'}")
    print("-" * 55)

    for idx, frame in enumerate(frames):
        if idx % i_frame_interval == 0:
            frame_type = 'I'
            compressed = compress_iframe(frame, idx)
            decoded    = decompress_iframe(compressed)
            REFERENCE_BUFFER.append(decoded)
            bitstream  = serialize_iframe(compressed)
            info       = ""
        else:
            compressed = compress_pframe(frame, idx)
            if compressed is not None:
                frame_type = 'P'
                decoded    = decompress_pframe(compressed)
                REFERENCE_BUFFER.append(decoded)
                bitstream  = serialize_pframe(compressed)
                info       = f"MVs: {len(compressed['motion_vectors'])}"
            else:
                frame_type = 'I(fallback)'
                compressed = compress_iframe(frame, idx)
                decoded    = decompress_iframe(compressed)
                REFERENCE_BUFFER.append(decoded)
                bitstream  = serialize_iframe(compressed)
                info       = ""

        compressed_frames.append(compressed)
        bitstreams.append(bitstream)
        print(f"{idx+1:<8} {frame_type:<14} {len(bitstream):>12}  {info}")

    total = sum(len(b) for b in bitstreams)
    print("-" * 55)
    print(f"Compression complete!  Total frame data: {total:,} bytes")

    # Build global bitstream with header
    global_bitstream = build_bitstream(
        bitstreams, len(frames), fps, width, height, i_frame_interval)
    print(f"Global bitstream size: {len(global_bitstream):,} bytes  "
          f"(header + {len(frames)} frames)")

    return compressed_frames, bitstreams, global_bitstream


def decompress_video(bitstreams_or_global, from_global=False):
    """
    Decompress video frames.

    Pass from_global=True and a single bytes object produced by compress_video
    to decode from the global bitstream (validates header, prints metadata).
    Otherwise pass a plain list of per-frame bitstreams as before.
    """
    global REFERENCE_BUFFER
    REFERENCE_BUFFER = []

    if from_global:
        bitstreams, meta = parse_bitstream(bitstreams_or_global)
        print(f"\nGlobal bitstream header:")
        print(f"  Version    : {meta['version']}")
        print(f"  Frames     : {meta['num_frames']}  ({meta['fps']} fps)")
        print(f"  Resolution : {meta['width']}x{meta['height']}")
        print(f"  I-interval : {meta['i_frame_interval']}")
        types = meta['frame_types']
        i_count = types.count('I'); p_count = types.count('P')
        print(f"  Frame types: {i_count} I-frames, {p_count} P-frames")
        print(f"  Index      : {' '.join(types)}")
    else:
        bitstreams = bitstreams_or_global

    decoded_frames = []
    print(f"\nDecompressing {len(bitstreams)} frames...")

    for bitstream in bitstreams:
        compressed = deserialize_frame(bitstream)
        if compressed['type'] == 'I':
            decoded = decompress_iframe(compressed)
        else:
            decoded = decompress_pframe(compressed)
        REFERENCE_BUFFER.append(decoded)
        decoded_frames.append(decoded)

    print("Decompression complete!")
    return decoded_frames