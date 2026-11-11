
import numpy as np
from numba import njit, prange # pyright: ignore[reportMissingTypeStubs]
from numba.extending import intrinsic
from numba import types
from llvmlite import ir as llvm_ir
from numba.core import cgutils
from src.engine.utils import l2_normalise

from typing import Annotated, TypeAlias
import numpy.typing as npt
Array2DF32: TypeAlias = Annotated[npt.NDArray[np.float32], (None, None)]
Array1DF32: TypeAlias  = Annotated[npt.NDArray[np.float32], (None)]
Array1DUInt8: TypeAlias = Annotated[npt.NDArray[np.uint8], (None)]
Array2DUInt8: TypeAlias = Annotated[npt.NDArray[np.uint8], (None, None)]
Array1DUInt64: TypeAlias = Annotated[npt.NDArray[np.uint64], (None)]
Array2DUInt64: TypeAlias = Annotated[npt.NDArray[np.uint64], (None, None)]

@intrinsic
def popcnt_u64(typingctx, val):
    """Hardware popcount for uint64 using LLVM intrinsic."""
    sig = types.uint64(types.uint64)

    def popcnt_u64_impl(context, builder, sig, args):
        [val] = args
        # Declare LLVM's ctpop intrinsic for i64
        llvm_i64 = val.type
        fnty = llvm_ir.FunctionType(llvm_i64, [llvm_i64])
        llvm_ctpop = cgutils.get_or_insert_function(
            builder.module, fnty, "llvm.ctpop.i64"
        )
        result = builder.call(llvm_ctpop, [val])
        return result

    return sig, popcnt_u64_impl

@intrinsic
def popcnt_u8(typingctx, val):
    """Hardware popcount for uint8 using LLVM intrinsic."""
    sig = types.uint8(types.uint8)

    def popcnt_u8_impl(context, builder, sig, args):
        [val] = args
        # Declare LLVM's ctpop intrinsic for i8
        llvm_i8 = val.type
        fnty = llvm_ir.FunctionType(llvm_i8, [llvm_i8])
        llvm_ctpop = cgutils.get_or_insert_function(
            builder.module, fnty, "llvm.ctpop.i8"
        )
        result = builder.call(llvm_ctpop, [val])
        return result

    return sig, popcnt_u8_impl

@njit(
    [
        "f4(u1[::1],u1[::1])",
        types.float32(
            types.Array(types.uint64, 1, "C", readonly=True),
            types.Array(types.uint64, 1, "C", readonly=True),
        ),
    ],
    fastmath=True,
    nogil=True,
    boundscheck=False,
    locals={
        "result": types.int64,
        "intersection": types.uint64,
        "dim": types.intp,
        "i": types.uint16,
    },
)
def bit_hamming(x, y):
    r"""Hamming distance for bit-packed binary vectors.

    Counts the number of differing bits between two uint64 arrays, where each
    byte contains 64 packed binary features. More efficient than standard
    Hamming for binary data.

    .. math::
        D(x, y) = \sum_i \text{popcount}(x_i \oplus y_i)

    Returns the total count of differing bits (not normalized).
    """
    result = 0
    dim = x.shape[0]

    for i in range(dim):
        intersection = x[i] ^ y[i]
        result -= popcnt_u64(intersection)
    return np.float32(result)


# get vector of short integers
# subtract 1.5 from everything
# l2 normalise -> divide by norm
# then do scalar product
def rabit_distance(q_data, q_queries):
    cqdata = q_data - 1.5
    cqqueries = q_queries-1.5

    cqdata = l2_normalise(cqdata)
    cqqueries = l2_normalise(cqqueries)

    all_sims = cqdata * cqqueries.T

    return np.argpartition(all_sims, 30)


@njit(fastmath=True, boundscheck=False, nogil=True)
def bsp(v, w):
    # TODO: counting the bits of v&w 
    result = (v & w)
    byte_array = result.view(np.uint8)
    total = 0
    for byte in byte_array.flat:
        total += popcnt_u8(byte)

    return total

@njit(fastmath=True, boundscheck=False, nogil=True)
def b2sp(v, w):
    lt = int(len(v)/2)
    # TODO: get (v⁺, v⁻) and (w⁺, w⁻)
    v_minus = v[:lt]
    v_plus = v[lt:]
    w_minus = w[:lt]
    w_plus = w[lt:]
    
    part1 = bsp(v_plus, w_plus) + bsp(v_minus, w_minus)
    part2 = bsp(v_plus, w_minus) + bsp(v_minus, w_plus)
    
    return part1 - part2


@njit(fastmath=True, boundscheck=False, nogil=True)
def b2sp_returns(set_a, sign_a, set_b, sign_b):
    # This gives the position of each set bit
    intersect = np.bitwise_and(set_a, set_b)

    # This tells us where the bits are bad
    baddies = np.bitwise_xor(sign_a, sign_b)
    set_baddies = np.bitwise_and(intersect, baddies)
    baddie_count = popcnt_u64(set_baddies)

    # This is the (goodness + badness) - 2 badness = goodness - badness = 0.5 b2sp
    return popcnt_u64(intersect) - (2 * baddie_count)