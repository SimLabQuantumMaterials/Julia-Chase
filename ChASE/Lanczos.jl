__precompile__()
module ChASE_Lanczos

MATVEC_counter = 0
function MATVEC_reset()
    global MATVEC_counter
    MATVEC_counter = 0
end

include("modules.jl")
include("Common.jl")

"""
does the lanczos procedure using matrix `A` as input. Finds `m` eigenvalues and
returns the estimate of the spectrum's upper bound, the eigenvalues and the Ritz
vectors. If `V` is a matrix of `size(A,1)×m`, the Lanczos vectors are saved.

```
function bare_lanczos!( A, m::UInt64, V = nothing )
```
"""
function bare_lanczos!( A, m::UInt64, V = nothing )

    global MATVEC_counter

    N = size(A,1)
    if (m > N || m < 1) error("m should be between 1 and size(A,1)") end
    if !isnothing(V)
        if size(V) != (N,m)
            error("V has to either be nothing (not saving vectors) or (N,m)")
        end
    end

    # output arrays (diagonal d and sub-diagonal e)
    d = Array{Float64,1}(undef,m)
    e = Array{Float64,1}(undef,m)

    cmplx_ary = ChASE_Common.iscomplex(A)
    v1 = ChASE_Common.random_vec( N, cmplx_ary )
    v0 = typeof(v1)(zeros(N))
    w = typeof(v1)(zeros(N))

    # initialize α and β as tridiagonal elements of resulting matrix
    if cmplx_ary
        α = 0.0 + 0.0im
        β = 0.0 + 0.0im
    else
        α = 0.0
        β = 0.0
    end

    # implementation of the actual Lanczos procedure, following e.g.
    # \url{https://en.wikipedia.org/wiki/Lanczos_algorithm}
    for k=1:m
        if !isnothing(V)
            # save v1 vectors
            V[:,k] = v1
        end

        w = A * v1
        α = adjoint(w) * v1
        w -= UniformScaling(α)*v1
        d[k] = real(α)
        if k==m
            break
        end
        w -= UniformScaling(β)*v0
        β = norm(w)
        w .*= 1.0/β
        e[k] = real(β)

        v1,v0 = v0,v1
        w,v1 = v1,w

        MATVEC_counter += 1

    end

    # tridiagonal matrix T given by d and e (diagonal / sub-diagonal)
    Λ, Z = LinearAlgebra.LAPACK.stegr!('V','I',d,e[1:m-1],0,0,1,m)
    b_sup = max(abs(Λ[1]), abs(Λ[end])) + abs(real(β))

    return Λ, Z, b_sup
end

"""
Implements the Lanczos procedure of `https://arxiv.org/pdf/1805.10121.pdf`,
Alg. 6. Needs the matrix `A`, the number of Lanczos vectors `k`, the number of
iterations `nvec`. Additionally needs the number of requested eigenvalues `nev`
and extra search space `nex`. If `approx==true`, the algorithm only returns an
estimate for the upper bound of the spectrum (this case is used if approximations
for the eigenvalues and eigenvectors are available, e.g. when solving a
correlated problem). `σ_gauss` controls the width of the spectral DOS estimates.

```
function lanczos( A, k::UInt64, nvec::UInt64, nev::UInt64, nex::UInt64;
                  approx::Bool = true, σ_gauss::Float64 = 1.0 )
```
"""
function lanczos( A, k::UInt64, nvec::UInt64, nev::UInt64, nex::UInt64;
                  approx::Bool = true, σ_gauss::Float64 = 1.0 )
    if approx
        Λ_tilde, Z, b_sup = bare_lanczos!( A, k, nothing )
        return b_sup
    end

    N = size(A,1)
    ctype = ChASE_Common.choose_complex_type( A )
    dos_weights = Array{Float64,2}(undef,nvec,k)
    dos_λ = Array{Float64,2}(undef,nvec,k)
    V = Array{ctype,2}(undef,N,nev+nex)
    V_tmp = Array{ctype,3}(undef,nvec,N,k)
    Λ_tmp = Array{Float64,2}(undef,nvec,k)
    VV_tmp = Array{ctype,2}(undef,N,k)

    b_sup_full = 0.0
    μ_1 = 0.0
    for j=1:nvec
        # TODO julia is strange. Cannot modify view of array through function?!
        Λ_tilde_j, Z_j, b_sup_j = bare_lanczos!( A, k, VV_tmp )
        V_tmp[j,:,:] = VV_tmp * Z_j
        Λ_tmp[j,:] = Λ_tilde_j
        dos_weights[j,:] .= abs.(Z_j[1,:]).^2
        dos_λ[j,:] .= Λ_tilde_j
        b_sup_full = max(b_sup_full,b_sup_j)
        if j==1
            μ_1 = Λ_tilde_j[1]
        else
            μ_1 = min(μ_1,Λ_tilde_j[1])
        end
    end
    integral = function(t::Float64)
            addition_one = ones(Float64,size(dos_λ))
            t_sc = t*addition_one
            to_min = dos_weights .* (
                0.5*addition_one + 0.5*erf.(
                    (t_sc-dos_λ) * (1.0/((√2)*σ_gauss))
                )
            )
            result = sum(to_min) - (nvec*(nev+nex)/N)
            return result
        end
    # TODO this algorithm for zero-finding is sub-optimal (it can fail!)
    μ_nevnex = find_zero(integral,μ_1)

    # fill V with V_tmp corresponding to Λ_tmp[:i] <= μ_nevnex
    counter::Int = 0
    for i=1:nvec,j=1:k
        if Λ_tmp[i,j] <= μ_nevnex && counter < size(V,2)
            counter += 1
            V[:,counter] .= V_tmp[i,:,j]
        end
    end
    # fill the rest of V with random vectors
    iscmplx = ChASE_Common.iscomplex(A)
    ctype = ChASE_Common.choose_complex_type(A)
    for i=counter+1:size(V,2)
        V[:,i] = ChASE_Common.random_vec(N,iscmplx)
    end

    # intersperse
    tmpvec = Array{ctype,1}(undef,size(A,1))
    for i=1:counter
        j::Int64 = div(i*(nev+nex),counter)
        ChASE_Common.swap_col!(V,Int(i),j)
    end

    return μ_1, μ_nevnex, b_sup_full, V

end

end
