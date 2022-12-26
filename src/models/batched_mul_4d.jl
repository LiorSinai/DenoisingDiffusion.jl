
"""
batched_mul(A, B) -> C

Batched matrix multiplication in 4D. Result has `C[:,:,k,l] == A[:,:,k,l] * B[:,:,k,l]`
"""
function batched_mul(A::AbstractArray{T,4}, B::AbstractArray{T,4}) where {T}
    if (size(A, 2) != size(B, 1)) || (size(A, 3) != size(B, 3)) || (size(A, 4) != size(B, 4))
        message = "A has dimensions $(size(A)) but B has dimensions $(size(B))"
        throw(DimensionMismatch(message))
    end
    new_A = reshape(A, size(A, 1), size(A, 2), :)
    new_B = reshape(B, size(B, 1), size(B, 2), :)
    C = batched_mul(new_A, new_B)
    new_C = reshape(C, (size(C, 1), size(C, 2), size(A, 3), size(A, 4)))
    new_C
end
