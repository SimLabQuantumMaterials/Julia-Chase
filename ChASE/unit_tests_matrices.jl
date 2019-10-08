__precompile__()
module unit_tests_matrices

include("modules.jl")
include("Common.jl")
include("RayleighRitz.jl")

function perturb_matrix( Q, mag )
    # perturb matrix `Q` with random numbers of magnitude `mag`.
    perturbation = UniformScaling(mag) * rand(eltype(Q),size(Q)[1],size(Q)[2])
    Q_new = Q + perturbation
    result = Matrix(qr(Q_new).Q[:,1:size(Q)[2]])
    return result
end

function random_matrix_known_eigenvalues( n::Int )
    Lambda_real = Array{Float64,1}(undef,n)
    for i=1:n
        Lambda_real[i] = n-i+1
    end
    Q = rand(Haar(2),n)
    return Q, Hermitian(Q * diagm(Lambda_real) * adjoint(Q))
end

function random_matrix_known_eigenvalues( n::Int, Lambda_real::Array{Float64,1} )
    Q = rand(Haar(2),n)
    return Q, Hermitian(Q * diagm(Lambda_real) * adjoint(Q))
end

function clement_matrix( n::Int )
    C = zeros(Float64,n,n)
    for i=2:n
        C[i,i-1] = √((n-i+1)*(i-1))
        C[i-1,i] = √((n-i+1)*(i-1))
    end
    return C
end

function run( n, nval, mag )
    # generate ranodm unitary matrix
    println("random unitary matrix")
    Q, A = random_matrix_known_eigenvalues( n )
    # perturb first few entries of Q
    println("perturb Q")
    @time Q_tilde = perturb_matrix(Q[:,1:nval],mag)

    println("RayleighRitz")
    @time V, Lambda = ChASE_RayleighRitz.solve(A,Q_tilde)

    println("Residuals")
    @time Res = ChASE_RayleighRitz.residuals(A,V,Lambda)

    println(Res)

end

end
