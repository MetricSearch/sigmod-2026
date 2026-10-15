import numpy as np
from numba import njit, prange
from engine.llvm_taps import *

# This version makes evp map to different nodes early on Mac and is inconsistent between Mac and Unix.
def get_evp_ternary_row_mix(row: np.ndarray) -> np.ndarray:
    row_len = len(row)
    x = int(np.ceil(row_len / 2))
    median_row = np.median(np.abs(row))
    n_row = np.zeros(row_len)

    pos_mask = row > median_row
    neg_mask = row < -median_row

    n_row[pos_mask] = 1.0
    n_row[neg_mask] = -1.0

    accounted = np.count_nonzero(n_row)

    if accounted<x:
        sorted_row = np.argpartition(row, x)
        for i in sorted_row:
            if n_row[i] == 0 and (accounted<x):
                n_row[i] = 1
                accounted+=1
            if accounted == x:
                break
    
    return n_row

# Consistent between Mac and Unix - but makes evp map to different nodes later on Mac
def get_evp_ternary_row_mask(row: np.ndarray) -> np.ndarray:
    row_len = len(row)
    x = int(np.ceil(row_len / 2))
    median_row = np.median(np.abs(row))
    n_row = np.zeros(row_len)

    pos_mask = row > median_row
    neg_mask = row < -median_row

    n_row[pos_mask] = 1.0
    n_row[neg_mask] = -1.0

    accounted = np.count_nonzero(n_row)

    if accounted<x:
        zero_indices = np.where(n_row == 0)[0]
        needed = x - accounted
        n_row[zero_indices[:needed]] = 1.0
    
    return n_row

# This version makes evp map to different nodes early on Mac and is inconsistent between Mac and Unix.
def get_evp_row_arg_partition(row: np.ndarray) -> np.ndarray:
    row_len = len(row)
    x = int(np.ceil(row_len / 2))
    median_row = np.median(np.abs(row))
    sorted_row = np.argpartition(row, x)
    n_row = np.zeros(row_len)
    accounted = 0

    for i in sorted_row:
        if row[i] > median_row:
            n_row[i] = 1
            accounted+=1
        elif row[i] < -median_row:
            n_row[i] = -1
            accounted+=1

    if accounted<x:
        for i in sorted_row:
            if n_row[i] == 0 and (accounted<x):
                n_row[i] = 1
                accounted+=1
            if accounted == x:
                break
    
    return n_row


# Consistent between Mac and Unix - makes evp map to different nodes earlier on Mac
def get_evp_row_arg_partition_stable(row: np.ndarray) -> np.ndarray:
    row_len = len(row)
    x = int(np.ceil(row_len / 2))
    sorted_row = np.argsort(row, kind="stable")[:x]
    n_row = np.zeros(row_len)
    accounted = 0

    for i in sorted_row:
        if row[i] > 0:
            n_row[i] = 1
            accounted += 1
        elif row[i] < 0:
            n_row[i] = -1
            accounted += 1

    if accounted < x:
        for i in sorted_row:
            if n_row[i] == 0 and (accounted < x):
                n_row[i] = 1
                accounted += 1
            if accounted == x:
                break

    return n_row


# Consistent between Mac and Unix - makes evp map to different nodes earlier on Mac
def get_evp_row_mix_stable(row: np.ndarray) -> np.ndarray:
    row_len = len(row)
    x = int(np.ceil(row_len / 2))
    median_row = np.median(np.abs(row))
    sorted_row = np.argsort(row, kind="stable")[:x]
    n_row = np.zeros(row_len)

    is_large_positive = row[sorted_row] > median_row
    is_large_negative = row[sorted_row] < -median_row

    n_row[sorted_row[is_large_positive]] = 1.0
    n_row[sorted_row[is_large_negative]] = -1.0

    accounted = np.count_nonzero(n_row)

    if accounted < x:
        for i in sorted_row:
            if n_row[i] == 0 and (accounted < x):
                n_row[i] = 1
                accounted += 1
            if accounted == x:
                break

    return n_row

def get_evp_ternary(data: np.ndarray) -> np.ndarray:
    d_res = []
    for i in range(len(data)):
        d_res.append(get_evp_ternary_row_mix(data[i]))
    return d_res

def get_evp_binary_row(row: np.ndarray):
    median_row = np.median(np.abs(row))
    d_plus_row = (row > median_row).astype(np.uint8)
    d_minus_row = (row < -median_row).astype(np.uint8)

    return d_minus_row, d_plus_row

#evp quantisation in v+ v- form
def get_evp_binary(data: np.ndarray) -> np.ndarray:
    d_plus = []
    d_minus = []
    for i in range(len(data)):
        d_minus_row, d_plus_row = get_evp_binary_row(data[i])
        d_plus.append(d_plus_row)
        d_minus.append(d_minus_row)

    d_minus = np.asarray(d_minus)
    d_plus = np.asarray(d_plus)
    
    evp_array = np.hstack([d_minus, d_plus])

    return evp_array

#evp quantisation in sign+ sign- set form
def get_evp_all(data: np.ndarray) -> np.ndarray:
    d_plus = []
    d_minus = []
    for i in range(len(data)):
        d_minus_row, d_plus_row = get_evp_binary_row(data[i])
        d_plus.append(d_plus_row)
        d_minus.append(d_minus_row)

    sign = np.asarray(d_plus, dtype=np.uint64)
    d_minus_var = np.asarray(d_minus, dtype=np.uint64)
    
    return sign, (d_minus_var+sign).astype(np.uint64)