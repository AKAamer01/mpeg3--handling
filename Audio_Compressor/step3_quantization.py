import numpy as np

def quantize_band(band_matrix, bits):
    if bits == 0: return np.zeros_like(band_matrix), 0
    levels = (2 ** bits) - 1
    max_val = np.max(np.abs(band_matrix))
    if max_val == 0: return band_matrix, 0
    
    real_q = np.round((np.real(band_matrix) / max_val) * levels)
    imag_q = np.round((np.imag(band_matrix) / max_val) * levels)
    return (real_q + 1j * imag_q), max_val

def bit_allocation_and_quantize(low, mid, high):
    l_q, l_max = quantize_band(low, 8)
    m_q, m_max = quantize_band(mid, 6)
    h_q, h_max = quantize_band(high, 4)
    return (l_q, l_max), (m_q, m_max), (h_q, h_max)

def dequantize_band(q_matrix, max_val, bits):
    levels = (2 ** bits) - 1
    return (np.real(q_matrix) / levels * max_val) + 1j * (np.imag(q_matrix) / levels * max_val)