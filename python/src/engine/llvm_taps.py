from numba.extending import overload, overload_method, intrinsic
from numba.core import cgutils
from numba.core.typing import signature
from numba import types
from llvmlite import ir


def extract_2bit_fast(word, pos):
    """LLVM intrinsic wrapper for optimized 2-bit extraction."""
    return (word >> (62 - 2 * pos)) & 3

@intrinsic
def typ_extract_2bit(typingctx, word, pos):
    """Type signature for the intrinsic"""
    sig = types.uint8(word, pos)
    
    def codegen(context, builder, sig, args):
        word_val, pos_val = args
        
        # Create the instruction sequence directly with CORRECT API
        
        # shift_amt = 62 - 2*pos
        two = ir.Constant(pos_val.type, 2)
        sixty_two = ir.Constant(pos_val.type, 62)
        pos_times_two = builder.mul(pos_val, two)
        shift_amt = builder.sub(sixty_two, pos_times_two)
        
        # shifted = word >> shift_amt (use lshr, not lshrdi)
        shifted = builder.lshr(word_val, shift_amt)
        
        # masked = shifted & 3
        three = ir.Constant(shifted.type, 3)
        masked = builder.and_(shifted, three)
        
        # Truncate to uint8
        return builder.trunc(masked, ir.IntType(8))
    
    return sig, codegen



@overload(extract_2bit_fast)
def impl_extract_2bit(word, pos):
    def impl(word, pos):
        return typ_extract_2bit(word, pos)
    return impl