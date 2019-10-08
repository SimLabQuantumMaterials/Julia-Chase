__precompile__()
module ChASE_RayleighRitz

MATVEC_counter = 0
function MATVEC_reset()
    global MATVEC_counter
    MATVEC_counter = 0
end

include("modules.jl")
include("Common.jl")

"""
hermitian problem matrix `A` and projection matrix `Q` as arguments.
returns eigenvectors `QW` and eigenvalues `Lambda` after solving reduced
eigenproblem (`adjoint(Q)AQ`)

```
function solve( A, Q )
```
"""
function solve( A, Q )
    global MATVEC_counter
    MATVEC_counter += size(Q,2)*2

    ChASE_Common.iscomplex(A) ? conjugate_char = 'C' : conjugate_char = 'T'
    ctype = ChASE_Common.choose_complex_type(A)

    tmp = zeros(ctype,size(Q,2),size(A,1))
    BLAS.gemm!(conjugate_char,'N',ctype(1.0),Q,Matrix(A),ctype(0.0),tmp)
    G = zeros(ctype,size(Q,2),size(Q,2))
    BLAS.gemm!('N','N',ctype(1.0),tmp,Q,ctype(0.0),G)

    solution = LinearAlgebra.eigen(ChASE_Common.check_hermitian(A)(G))
    Lambda = solution.values
    W = solution.vectors

    return Q*W, Lambda
end

"""
calculates the residuals of the eigenproblem of matrix `A` (needs `A`,
eigenvectors `V` and eigenvalues `Lambda`) i.e. returns `A V - V diagm(Lambda)`
which contains all residuals column-wise.

```
function residuals( A, V, Lambda )
```
"""
function residuals( A, V, Lambda )
    global MATVEC_counter
    res = A * V - V * diagm(Lambda)
    MATVEC_counter += size(V,2)*2
    return [norm(res[:,i]) for i in range(1,stop=size(res,2))]
end

end
