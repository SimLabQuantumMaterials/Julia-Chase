__precompile__()
module ChASE_Filter

MATVEC_counter = 0
function MATVEC_reset()
    global MATVEC_counter
    MATVEC_counter = 0
end

include("modules.jl")
include("Common.jl")

"""
Chebychev filter. Takes the approximate eigenvectors `V` and filters them
according to the eigenvalue `λ_1` and the bounds `lower` and `upper` with the
degrees array `degrees`. The matrix `V` is modified during this procedure.
Returns the filtered matrix `W`. See `https://arxiv.org/pdf/1805.10121.pdf`,
Alg. 4.

```
function filter!( A, V, λ_1::Float64, lower::Float64, upper::Float64,
                 degrees::Array{UInt64,1} )
```
"""
function filter!( A, V, λ_1::Float64, lower::Float64, upper::Float64,
                 degrees::Array{UInt64,1} )
    global MATVEC_counter

    if lower>=upper
        @warn "lower bound larger than upper bound. inverting."
        upper,lower = lower,upper
    end
    if size(degrees,1) != size(V,2)
        @warn "degree array not the size of V" size(degrees,1) size(V,2)
    end
    for i=1:size(degrees,1)
        if degrees[i] % 2 != 0
            d = Int(degrees[i])
            error("degree $i not even ($d) – returning wrong result")
        end
    end

    c = 0.5 * (upper + lower)
    e = 0.5 * (upper - lower)
    σ_1 = e / (λ_1 - c)

    hermitian = ChASE_Common.check_hermitian(A)
    ctype = ChASE_Common.choose_complex_type(A)
    cshift = hermitian(UniformScaling(c)*Matrix{ctype}(I,size(A)))
    A -= cshift

    α = (ctype(σ_1 / e))
    W = BLAS.gemm('N','N',α,Matrix(A),V)
    MATVEC_counter += size(V,2)

    σ = σ_1
    s = Int64(1)

    for t=2:degrees[end]
        τ = 1.0 / (2.0 / σ_1 - σ)
        α = (ctype(2.0*τ / e))
        β = (ctype(σ*τ))
        MATVEC_counter += (size(W,2)-s)
        #V[:,s:end] .= α * A * W[:,s:end] - β * V[:,s:end]
        # TODO unintitive BLAS call (the ! has no consequence if using a view)
        V[:,s:end] = BLAS.gemm!( 'N', 'N', α, Matrix(A), W[:,s:end], -β, V[:,s:end])
        V, W = W, V
        σ = τ
        while (degrees[s] <= t && s<size(degrees,1))
            s += 1
        end
    end

    A += cshift

    return W

end

"""
Degree optimization corresponding to eigenvalue `λ` in interval `(c,e)`,
tolerance `tol` and residual `residual`. The extra degree, `deg_extra` is added
for safety. See `https://arxiv.org/pdf/1805.10121.pdf`, Alg. 5.

```
function degrees( tol::Float64, residual::Float64, λ, c, e; deg_extra::UInt64 = 2,
                 deg_max::UInt64 = 36 )
```
"""
function degrees( tol::Float64, residual::Float64, λ, c, e; deg_extra::UInt64 = 2,
                 deg_max::UInt64 = 36 )
    t = (λ - c)/e
    rho = max( abs(t-√Complex(t^2-1)), abs(t+√Complex(t^2-1)) )
    xval = abs(log(residual/tol)/log(rho))
    if xval == Inf
        deg = deg_max
    else
        try
            deg = Int64(ceil( abs( log( residual/tol ) / log(rho) ) ))
            deg = min(deg+deg_extra,deg_max)
        catch InexactError
            deg = deg_max
        end
    end
    # force even degrees
    return deg + deg % 2
end

end
