using Test

test_rr = true
test_filter = true
test_common = true
test_lanczos = false

if test_rr begin
# unit tests - module `ChASE_RayleighRitz.jl`
include("RayleighRitz.jl")
include("unit_tests_matrices.jl")

using LinearAlgebra

A_test = unit_tests_matrices.clement_matrix( 100 )
sol_test = ChASE_RayleighRitz.eigen(A_test)
λ_test = sol_test.values
v_test = sol_test.vectors
RR_test = ChASE_RayleighRitz.solve( A_test, v_test[:,1:15] )

# real
@test RR_test[2] ≈ λ_test[1:15] atol = 1.e-12
@test abs.(adjoint(RR_test[1]) * v_test[:,1:15]) - I ≈ zeros((15,15)) atol = 1.e-10

Q_test, C_test = unit_tests_matrices.random_matrix_known_eigenvalues( 100 )
RR_ctest = ChASE_RayleighRitz.solve( C_test, Q_test[:,1:15] )

# complex
@test RR_ctest[2] ≈ [Float64(100-15+i) for i in 1:15] atol = 1.e-12

# residuals
@test ChASE_RayleighRitz.residuals( A_test, RR_test[1], RR_test[2] ) ≈ zeros(15) atol = 1.e-10
@test ChASE_RayleighRitz.residuals( C_test, RR_ctest[1], RR_ctest[2] ) ≈ zeros(15) atol = 1.e-11

# some typing / dimension mismatch / etc
@test_throws MethodError ChASE_RayleighRitz.solve( 1.0, 2.0 )
@test_throws MethodError ChASE_RayleighRitz.solve( "yos", "nos" )
@test_throws DimensionMismatch ChASE_RayleighRitz.solve( zeros(Float64,(100,200)) + I, 1.0 )
@test_throws DimensionMismatch ChASE_RayleighRitz.solve( zeros(Float64,(100,100)) + I,
                                                        zeros(Float64,(15,15)) + I )

# highly degenerate problem
λ_deg_test = [Float64(10) for i in 1:100]
λ_deg_test[50:end] .= [Float64(20) for i in 50:100]
B_test = Hermitian(Q_test * diagm(λ_deg_test) * adjoint(Q_test))
RR_deg_test = ChASE_RayleighRitz.solve( B_test, Q_test[:,40:60] )

# eigen, residuals, unitarity
@test RR_deg_test[2] ≈ λ_deg_test[40:60] atol = 1.e-12
@test ChASE_RayleighRitz.residuals( B_test, RR_deg_test[1], RR_deg_test[2] ) ≈ zeros(21) atol = 1.e-12
@test adjoint(RR_deg_test[1]) * RR_deg_test[1] - I ≈ zeros((21,21)) atol = 1.e-12
end end # begin # test_rr

if test_filter begin
# unit tests - module `ChASE_Filter.jl`
include("unit_tests_matrices.jl")
using LinearAlgebra

# defining chebychev polynomials
function chebychev( x, degree )
    if degree == 1 return x elseif degree == 0
        if typeof(x) <: Matrix{<:Number} || typeof(x) <: Hermitian{<:Number} ||
           typeof(x) <: Symmetric{<:Number}
            return I else return 1 end
    else return 2*x*chebychev(x,degree-1)-chebychev(x,degree-2) end
end

# testing chebychev polynomials with random matrix
cmat = Matrix{Float64}(undef,100,100)
for i=1:100 for j=1:100 cmat[i,j] = rand() end end
@test chebychev(cmat,3) ≈ 2*cmat*(2*cmat*cmat-I)-cmat atol = 1.e-12

# testing the filter routine against manual chebychev multiplication
include("Filter.jl")
include("Common.jl")
dim = 500; cheby = 8; fdim = 12
Λ = Array{Float64,1}(undef,dim); for i=1:dim Λ[i] = Float64(i)/Float64(dim) end
degrees = Array{UInt64,1}(undef,fdim); for i=1:fdim degrees[i] = cheby end
Q, A = unit_tests_matrices.random_matrix_known_eigenvalues(dim,Λ)
Q = Q[:,1:fdim]
C_man = chebychev(A,cheby) * Q
C_fil = ChASE_Filter.filter!(A, Q, 1.0, -1.0, 1.0, degrees)
@test C_man≈C_fil atol=1.e-12

# testing the filter routine for different degrees in degree array
degrees = Array{UInt64,1}(undef,fdim)
cheby = 2
sq = size(Q)
# random matrix
Q = Matrix{ComplexF64}(undef,sq)
for i=1:sq[1] Q[i,:] = ChASE_Common.random_vec( sq[2], true ) end
for i=1:div(fdim,3) degrees[i] = cheby end
for i=div(fdim,3):2*div(fdim,3) degrees[i] = cheby*2 end
for i=2*div(fdim,3):fdim degrees[i] = cheby*3 end
C_manual = Matrix{ComplexF64}(undef,dim,fdim)
for i in 1:fdim C_manual[:,i] = chebychev(A,degrees[i]) * Q[:,i] end
C_filter = ChASE_Filter.filter!(A, Q, 1.0, -1.0, 1.0, degrees)
@test C_manual≈C_filter atol=1.e-12

# testing whether odd degrees throw an error
odd_degrees = Array{UInt64,1}([2 for i=1:fdim])
odd_degrees[div(fdim,2)] = 3
@test_throws ErrorException ChASE_Filter.filter!(A, Q, 1.0, -1.0, 1.0, odd_degrees )

end end # begin # test_filter

if test_lanczos begin
# unit tests - module `ChASE_Lanczos.jl`
include("Lanczos.jl")

end end # begin # test_lanczos

if test_common begin
# unit tests - module `ChASE_Common.jl`
include("Common.jl")

using LinearAlgebra

# square matrix, generate orthogonal components
A = zeros(ComplexF64,(200,200))
A += I
for i=101:200 A[:,i] = ChASE_Common.random_vec(200, true) end
Q = ChASE_Common.fast_qr!( A )
is_ortho = Array{Float64,1}(undef,1)
is_ortho[1] = 0.0
for i=101:200
    vec = Q[:,i]
    is_ortho[1] += sum(abs.(vec[1:100]))
end
@test is_ortho[1] ≈ 0.0 atol = 1.e-12

# nonsquare matrix, generate orthogonal components
A = zeros(ComplexF64,(400,200))
A[1:100,1:100] += I
for i=101:200 A[:,i] = ChASE_Common.random_vec(400, true) end
Q = ChASE_Common.fast_qr!( A )[:,1:200]
is_ortho = Array{Float64,1}(undef,1)
is_ortho[1] = 0.0
for i=101:200
    vec = Q[:,i]
    is_ortho[1] += sum(abs.(vec[1:100]))
end
for i=102:200
    @test adjoint(Q[:,101]) * Q[:,i] ≈ 0.0 atol = 1.e-12
    @test adjoint(Q[:,99]) * Q[:,i] ≈ 0.0 atol = 1.e-12
end
@test is_ortho[1] ≈ 0.0 atol = 1.e-12
@test det(adjoint(Q) * Q - I) ≈ 0.0 atol = 1.e-12

end end # begin # test_common

# unit tests - `ChASE.jl`
include("../ChASE.jl")
