import numpy as np

def rle_encode(data):
    flat = np.asarray(data).flatten()
    if len(flat) == 0: return []
    encoded, prev, count = [], flat[0], 1
    for val in flat[1:]:
        if val == prev: count += 1
        else:
            encoded.append((prev, count))
            prev, count = val, 1
    encoded.append((prev, count))
    return encoded

def rle_decode(encoded, shape):
    decoded = []
    for val, count in encoded: decoded.extend([val] * count)
    return np.array(decoded).reshape(shape)