import time
from numpy.linalg import norm
import numpy as np
import pandas as pd
from pathlib import Path
import heapq


# Original rabit quantisation logic
def rRound_original(datum, t):
    # Ensure datum is a numpy array for vectorised operations
    scalar = datum.astype(np.float64)* t
    rounded = np.floor(scalar) + 0.5
    return rounded

def next_critical_original(o_prime, y_cur, t):

    crits = (y_cur + 0.5) / o_prime
    # Filter for values strictly greater than current_t
    new_crits = crits[crits > t]
    
    return new_crits


@njit(parallel=False, nogil=True, boundscheck=False, fastmath=True)
def rRound_c_version(datum, t):
    # Ensure datum is a numpy array for vectorised operations
    scalar = datum.astype(np.float64) * t
    rounded = np.zeros(len(scalar))

    # negative map fix (me, ben and richard talk)
    for i in range(len(scalar)):
        magnitude = scalar[i]
        if magnitude<0:
            rounded[i] = -(np.round(-magnitude) + 0.5)
        else:
            rounded[i] = np.round(magnitude) + 0.5
    return rounded



@njit(parallel=False, nogil=True, boundscheck=False, fastmath=True)
def next_critical_c_version(o_prime, y_cur, t):

    crits = (y_cur[o_prime!=0] + 0.5) / o_prime[o_prime!=0]
    # Filter for values strictly greater than current_t
    new_crits = crits[crits > t]
    
    return new_crits

@njit(parallel=False, nogil=True, boundscheck=False, fastmath=True)
def get_rabit_c_version(o_prime):

    # Initialise variables
    sp = 0.0
    mag = 0.0
    v_max = 0.0
    t_max = 0.0

    # if (len(o_prime)>50):
    #     t = 1e-10  # 0.0000000001
    # else:
    #     t = 1

    # t = 1e-5

    max_o = np.max(np.abs(o_prime))

    t_end = float(3 + 10) / max_o

    t = 0.052 * t_end

    crits = List.empty_list(np.float64)
    heapq.heapify(crits)
        
    o_prime = np.asarray(o_prime, dtype=np.float64)
        
    # Initial rounding
    y_cur = rRound(o_prime, t)

    sp = np.dot(y_cur, o_prime.T)
    # Compute magnitude of y_cur
    mag = np.linalg.norm(y_cur)

    v_cur = sp / mag
                    
    # Avoid division by zero
    if v_cur > v_max:
        v_max = v_cur
        t_max = t
        
    while True:
            # Get next critical points
            next_crits = next_critical_new(o_prime, y_cur, t)
            # If there are critical points, push them into the heap
            if next_crits.size > 0:
                for c in next_crits:
                    if c not in crits:
                        heapq.heappush(crits, c)
        
            if len(crits) == 0:
                break
        
            t = heapq.heappop(crits)
                    
            # Compute next rounded vector
            y_cur = rRound(o_prime, t)
            # Check termination condition
            if np.max(np.abs(y_cur)) > 2:
                break
            else:
                # Compute sp = y_cur * o_prime' (dot product)
                sp = np.dot(y_cur, o_prime.T)
                # Compute magnitude of y_cur
                mag = np.linalg.norm(y_cur)
        
                v_cur = sp / mag
                        
                # Avoid division by zero
                if v_cur > v_max:
                    v_max = v_cur
                    t_max = t
        
    # Final result
    y_bar = rRound(o_prime, t_max)
    mag = np.linalg.norm(y_bar)
    yu_bar = y_bar + 1.5  # (3/2) = 1.5
        
    return yu_bar, mag



# Reverse engineered version of theirs:
# rounding to the closest point is what we're looking for so steps should be:
# Round it
# The 0.5 correction doest work in the bottom left quarter (the all small negative)
# The opposite of it doesn't work in the top right quarter (the all small positive)
# Doing both results in the same exact issue but both ways (plus we have big issues with shapes)
# So the rounding is wrong
# Now what to do:
# 1- Smoke crack
# 2- Read paper
# 3- hallucinate (not the gpt version)
# 4- read some more (because surely they have said something about rounding)
# 5- Huzzah, I can reverse engineer their code from their repo this is what they do:
# Use number of bits to fill stuff in (clipper->3=2^2-1)
# If we are past the next integer: double up
# If we are below the current integer boundary: double down
def kRound(datum, t, clipper=3):
    # multiply datum by t as saud in paper
    scalar = datum.astype(np.float64)* t

    #Limit the numbers to n_bits
    clipped = np.clip(scalar, 0, clipper)

    code = np.floor(clipped).astype(np.int64)

    bounded = code < clipper

    safe_datum = np.where(datum>0, datum.astype(np.float64), 0)
    
    # If we are past the next integer: double up
    double_up = bounded & ((code + 1.0)/safe_datum <= t)

    # If we are below the current integer boundary: double down
    double_down = (~double_up) & (code > 0) & ((code / safe_datum) > t)

    adjust = np.where(double_up, 1, np.where(double_down, -1, 0)).astype(np.int64)
    
    code = code + adjust.astype(np.int64)

    for i in range(len(datum)):
        if datum[i] == 0:
            code[i] = 0

    return code

def next_critical_github(o_prime, y_cur, t_end, clipper=3):

    # Compute critical values: (y_cur(i) + 1) / o_prime(i)
    inbounds = (y_cur < clipper) & (o_prime > 0)
    
    if not np.any(inbounds):
        return np.array([])
    
    crits = (y_cur[inbounds] + 1.0) / o_prime[inbounds]
    # Filter for values strictly greater than current_t
    new_crits = crits[crits < t_end]
    
    return new_crits

def get_t(o_prime, max_no):
    # Initialise variables
    sp = 0.0
    mag = 0.0
    v_max = 0.0
    t_max = 0.0

    t_start = 1e-10  # 0.0000000001

    max_o = np.max(o_prime)
    if max_o == 0:
        return np.zeros_like(o_prime)  # zero vector in, zero out
    
    t_end = float(max_no + 10) / max_o

    # t_start = t_end*0.2
    
    t = t_start
    crits = []
    heapq.heapify(crits)
        
    o_prime = np.asarray(o_prime, dtype=np.float64)

    # Initial rounding
    y_cur = kRound(o_prime, t, max_no)

    sp = np.dot(y_cur+0.5, o_prime.T)
    # Compute magnitude of y_cur
    # mag = np.linalg.norm(y_cur)
    
    mag = np.sqrt(float(o_prime.size) * 0.25 + (np.sum(y_cur**2 + y_cur)))
                    
    # Avoid division by zero
    v_val = sp / mag
    
    if v_val > v_max:
        v_max = v_val
        t_max = t

    # while some critical values have not been enumerated
    while True:

        # Get next critical points
        next_crits = next_critical_github(o_prime, y_cur, t_end, max_no)

        # If there are critical points, push them into the heap
        if next_crits.size > 0:
            for c in next_crits:
                if c <= t_end and c not in crits:
                    heapq.heappush(crits, c)

        if len(crits) == 0:
            break

        # Update 𝑡 with the next smallest critical value
        t = heapq.heappop(crits)

        # Update y𝑐𝑢𝑟,⟨y𝑐𝑢𝑟,o′⟩and ∥y𝑐𝑢𝑟∥ with the new 𝑡
        # Compute next rounded vector
        y_cur = kRound(o_prime, t, max_no)

        # Compute sp = y_cur * o_prime' (dot product)
        # sp = np.dot(y_cur+0.5, o_prime.T)
        # Compute magnitude of y_cur
        mag = np.linalg.norm(y_cur)

        mag = np.sqrt(float(o_prime.size) * 0.25 + (np.sum(y_cur**2 + y_cur)))

        # if ⟨y𝑐𝑢𝑟,o′⟩/∥y𝑐𝑢𝑟∥> 𝑣𝑚𝑎𝑥        
        # Avoid division by zero
        v_val = sp / mag

        if v_val > v_max:
            v_max = v_val
            t_max = t

    # Final result
    return t_max

def get_rabit_from_github(o_prime):
    n_bits = 2
    max_no = (1 << n_bits) - 1

    t_max = get_t(o_prime, max_no)

    # y_bar = o_prime * t_max
    y_bar = kRound(o_prime, t_max)
    yu_bar = y_bar + max_no/2  # (3/2) = 1.5

    return yu_bar
