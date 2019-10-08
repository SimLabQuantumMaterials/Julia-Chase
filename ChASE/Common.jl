__precompile__()
module ChASE_Common

include("modules.jl")

"""
struct for the precondition used in ChASE. Usage only possible if both `V` and
`λ` are set accordingly
"""
@with_kw mutable struct precond
    V = nothing
    λ = nothing
end

"""
struct to handle output verbosity / debugging options. Contains counter
switches (`counter_matvec`, `counter_while` and `counter_per_step`) as well as
debug switches (`debug_residuals`, `debug_eigenvalues`, `debug_degree_optim`,
`debug_bounds`).
"""
@with_kw mutable struct verbosity
    counter_matvec::Bool = false
    counter_while::Bool = false
    counter_per_step::Bool = false

    debug_residuals::Bool = false
    debug_eigenvalues::Bool = false
    debug_degree_optim::Bool = false
    debug_bounds::Bool = false
end

"""
main configuration struct.
`approx` should be either `nothing` or of type `precond` to indicate the usage
of an approximate solution to a correlated eigenproblem.
`tol` sets the tolerance of the residuals
`max_iter` sets the maximum iterations performed in the main `while` loop of the
algorithm
`optim` enables chebychev degree optimization
`deg_extra` sets the additionally added degree if `optim == true`
`deg` sets the initial degree (which is used throughout the algorithm if
`optim==false`)
`lanczos_iter` sets the number of lanczos steps to be performed per procedure
`lanczos_num` sets the number of lanczos procedures performed to do the DOS
estimation
`lanczos_dos_σ` sets the standard deviation of the gaussians used in the
spectral DOS estimate
"""
@with_kw mutable struct config
    approx = nothing
    tol::Float64 = 1.e-10
    max_iter::UInt64 = 25
    optim::Bool = false
    deg_extra::UInt64 = 2
    deg_max::UInt64 = 36
    deg::UInt64 = 20
    lanczos_iter::UInt64 = 25
    lanczos_num::UInt64 = 4
    lanczos_dos_σ::Float64 = 1.0
end

"""
struct to define matrix inputs from files. The `filename` parameter is mandatory
and has to be the full filename. `filetype` can be one of `text`, `bin` or
`hdf5`.
`is_complex` is needed for text or binary input
`datasetname` is needed for hdf5 input
`matrix_size` is needed for binary input
`is_square` is needed for binary input
`matrix_size_2` is needed for binary input in case the matrix is not square. It
indicates the second (column) dimension.
`tolerance_for_checks` defines what is still considered a hermitian matrix.
"""
@with_kw mutable struct matrix_input
    filename::AbstractString
    filetype::AbstractString = "text"
    is_complex::Bool = true
    datasetname::AbstractString = "A"
    matrix_size = nothing
    is_square::Bool = true
    matrix_size_2 = nothing # if non-square, read (matrix_size, matrix_size_2)
    tolerance_for_checks::Float64 = 1.e-5
end

"""
similar struct to `matrix_input`. The `filename` parameter is mandatory and has
to be the full filename. `filetype` can be one of `text`, `bin` or `hdf5`.
`is_complex` is needed for text or binary input
`datasetname` is needed for hdf5 input
`vector_size` is needed for binary input
"""
@with_kw mutable struct vector_input
    filename::AbstractString
    filetype::AbstractString = "text"
    is_complex::Bool = false
    datasetname::AbstractString = "v"
    vector_size::UInt64 = nothing
end

"""
function to read matrices from files defined through the `matrix_input` struct.
Additionally checks whether the given matrix is hermitian if `check_hermitian`
is set.

```
function read_matrix( input::matrix_input; check_hermitian::Bool = true )
```
"""
function read_matrix( input::matrix_input; check_hermitian::Bool = true )
    filetype = input.filetype
    filename = input.filename
    if filetype == "text"
        is_complex = input.is_complex
        result = read_matrix_text( filename, is_complex )
    elseif filetype == "hdf5"
        datasetname = input.datasetname
        result = read_matrix_hdf5( filename, datasetname )
    elseif filetype == "bin"
        matrix_size = input.matrix_size
        is_complex = input.is_complex
        if isnothing(matrix_size)
            error("filetype=bin needs matrix_size")
        end
        if input.is_square
            result = read_matrix_binary( filename, UInt64(matrix_size),
                                                      is_complex )
        else
            if isnothing(input.matrix_size_2)
                error("for non-square matrix, second dimension must be given")
            end
            result = read_matrix_binary( filename, UInt64(matrix_size),
                                                      is_complex; is_square = input.is_square,
                                                      matrix_size_2 = UInt64(input.matrix_size_2) )
        end
    else
        error("specify filetype as text, bin or hdf5")
    end
    if check_hermitian
        tol = input.tolerance_for_checks
        if input.is_complex
            sumabs = sum(abs.(result-adjoint(result)))
            if isapprox(sumabs,0.0; atol=tol)
                return Hermitian(result)
            else
                @warn "matrix not hermitian, using A+A^†"
                return Hermitian(result + adjoint(result))
            end
        else
            if isapprox(result,transpose(result); atol=tol)
                return Symmetric(result)
            else
                @warn "matrix not symmetric, using A+A^T"
                return Symmetric(result + transpose(result))
            end
        end
    else
        return result
    end
end

"""
function to read vectors from files defined through the `vector_input` struct

```
function read_vector( input::vector_input )
```
"""
function read_vector( input::vector_input )
    filetype = input.filetype
    filename = input.filename
    if filetype == "text"
        is_complex = input.is_complex
        result = read_vector_text( filename, is_complex )
    elseif filetype == "hdf5"
        datasetname = input.datasetname
        result = read_vector_hdf5( filename, datasetname )
    elseif filetype == "bin"
        vector_size = input.vector_size
        is_complex = input.is_complex
        if isnothing(vector_size)
            error("filetype=bin needs vector_size")
        end
        result = read_vector_binary( filename, UInt64(vector_size),
                                                  is_complex )
    else
        error("specify filetype as text, bin or hdf5")
    end
    return result
end

"""
generate the `precond` object from `matrix_input` / `vector_input` objects or
julia arrays.
`V` and `λ` refer to the previous solution's eigenvectors / eigenvalues.

```
function generate_precondition( V, λ; checks_tolerance = 1.e-5 )
```
"""
function generate_precondition( V, λ; checks_tolerance = 1.e-5 )
    pre = precond()
    if typeof(V) <: Matrix{<:AbstractFloat} || typeof(V) <: Matrix{<:Complex{<:AbstractFloat}}
        pre.V = V
    elseif typeof(V) == matrix_input
        pre.V = read_matrix( V, check_hermitian = false )
    else
        error("unsupported type of matrix input V")
    end
    if typeof(λ) <: Vector{<:AbstractFloat}
        pre.λ = λ
    elseif typeof(λ) == vector_input
        pre.λ = read_vector( λ )
    else
        error("unsupported type of vector input λ (must be real)")
    end
    return pre
end

"""
check whether a matrix is hermitian or symmetric and return the corresponding
type

```
function check_hermitian( mat )
```
"""
function check_hermitian( mat )
    # check whether mat is sym / herm and return type conversion / warning +
    # empty lambda if not.
    if LinearAlgebra.ishermitian(mat)
        return Hermitian
    elseif LinearAlgebra.issymmetric(mat)
        return Symmetric
    else
        @warn "matrix neither hermitian nor symmetric"
        return x->x
    end
end

"""
check whether a matrix contains complex elements

```
function iscomplex( mat )
```
"""
function iscomplex( mat )
    if typeof(mat) <: Array{<:Real,2}
        return false
    else
        return true
    end
end

"""
return element type of a matrix depending on whether it's complex or real

```
function choose_complex_type( mat )
```
"""
function choose_complex_type( mat )
    if iscomplex( mat )
        return ComplexF64
    else
        return Float64
    end
end

"""
QR decomposition using LAPACK routines, faster than internal routine if only
`Q` is needed. Returns `Q` to a given matrix `A` and modifies `A`.

```
function fast_qr!( A )
```
"""
function fast_qr!( A )
    H,τ = LAPACK.geqrf!(A)
    Q = zeros(choose_complex_type(A),size(A,1),size(A,1))
    Q += I
    LAPACK.ormqr!('R','N',H,τ,Q)
    return Q
end

"""
swap the columns `i` and `j` of a matrix `A`

```
function swap_col!( A, i::Int, j::Int )
```
"""
function swap_col!( A, i::Int, j::Int )
    tmpvec = copy(A[:,i])
    A[:,i] .= copy(A[:,j])
    A[:,j] .= copy(tmpvec)
    return nothing
end

"""
swap the columns `i` and `j` of a matrix `A` using an already allocated vector
`tmpvec`

```
function swap_col!( A, tmpvec, i::Int, j::Int )
```
"""
function swap_col!( A, tmpvec, i::Int, j::Int )
    tmpvec[:] .= copy(A[:,i])
    A[:,i] .= copy(A[:,j])
    A[:,j] .= copy(tmpvec)
    return nothing
end

"""
return normed vector of size `N`, switch whether `complex` / real

```
function random_vec( N::Int, complex::Bool )
```
"""
function random_vec( N::Int, complex::Bool )
    if complex
        v = Array{ComplexF64,1}(UndefInitializer(),N)
        v = rand(Normal(),N) .+ im .* rand(Normal(),N)
    else
        v = Array{Float64,1}(UndefInitializer(),N)
        v = rand(Normal(),N)
    end
    v .*= 1.0/norm(v)
    return v
end

"""
read matrix from a binary dump. `matrix_size` refers to the first dimension,
in case `is_square == true` also to the second. Otherwise, `matrix_size_2` is
the column dimension. `is_complex` is mandatory and switches between `ComplexF64`
and `Float64`.

```
function read_matrix_binary( filename::AbstractString, matrix_size::UInt64, is_complex::Bool;
                             is_square::Bool = true, matrix_size_2::UInt64 = UInt64(1) )
```
"""
function read_matrix_binary( filename::AbstractString, matrix_size::UInt64, is_complex::Bool;
                             is_square::Bool = true, matrix_size_2::UInt64 = UInt64(1) )
    if is_complex
        ctype = ComplexF64
    else
        ctype = Float64
    end
    if is_square
        data = Array{ctype,1}(undef,matrix_size*matrix_size)
    else
        data = Array{ctype,1}(undef,matrix_size*matrix_size_2)
    end
    stream = open(filename,"r")
    read!(stream,data)
    close(stream)
    if is_square
        return reshape(data,(matrix_size,matrix_size))
    else
        return reshape(data,(matrix_size,matrix_size_2))
    end
end

"""
read matrix from a textfile, first define whether it is complex or not.

```
function read_matrix_text( filename::AbstractString, is_complex::Bool )
```
"""
function read_matrix_text( filename::AbstractString, is_complex::Bool )
    if is_complex
        dtype = ComplexF64
    else
        dtype = Float64
    end
    result = DelimitedFiles.readdlm(filename, '\t', dtype, '\n')
    if typeof(result) <: Matrix
        return result
    else
        error("not a matrix")
    end
end

"""
read matrix from a hdf5 file with a given datasetname

```
function read_matrix_hdf5( filename::AbstractString, datasetname::AbstractString )
```
"""
function read_matrix_hdf5( filename::AbstractString, datasetname::AbstractString )
    data = load(fname)
    result = data[datasetname]
    if typeof(result) <: Matrix
        return result
    else
        error("not a matrix")
    end
end

"""
read a vector from a binary file, `vector_size` and `is_complex` are both
mandatory.

```
function read_vector_binary( filename::AbstractString, vector_size::UInt64, is_complex::Bool )
```
"""
function read_vector_binary( filename::AbstractString, vector_size::UInt64, is_complex::Bool )
    if is_complex
        ctype = ComplexF64
    else
        ctype = Float64
    end
    data = Array{ctype,1}(undef,vector_size)
    stream = open(filename,"r")
    read!(stream,data)
    close(stream)
    return data
end

"""
read a (complex) vector from a text file (column)

```
function read_vector_text( filename::AbstractString, is_complex::Bool )
```
"""
function read_vector_text( filename::AbstractString, is_complex::Bool )
    if is_complex
        dtype = ComplexF64
    else
        dtype = Float64
    end
    result = DelimitedFiles.readdlm(filename, '\t', dtpye, '\n')
    if typeof(result) <: Vector
        return result
    else
        error("not a vector")
    end
end

"""
read a vector from an hdf5 file and a given datasetname

```
function read_vector_hdf5( filename::AbstractString, datasetname::AbstractString )
```
"""
function read_vector_hdf5( filename::AbstractString, datasetname::AbstractString )
    data = load(fname)
    result = data[datasetname]
    if typeof(result) <: Vector
        return result
    else
        error("not a vector")
    end
end

"""
fill configuration structs `cfg::config` and `switches::verbosity` from
.ini-style config file `filename`

```
function get_config!( filename::String, switches::verbosity, cfg::config )
```
"""
function get_config!( filename::String, switches::verbosity, cfg::config )
    conf = ConfParse(filename)
    parse_conf!(conf)
    try switches.counter_matvec = retrieve(conf, "counter", "matvec", Bool) catch KeyError end
    try switches.counter_while = retrieve(conf, "counter", "while", Bool) catch KeyError end
    try switches.counter_per_step = retrieve(conf, "counter", "per_step", Bool) catch KeyError end

    try switches.debug_residuals = retrieve(conf, "debug", "residuals", Bool) catch KeyError end
    try switches.debug_eigenvalues = retrieve(conf, "debug", "eigenvalues", Bool) catch KeyError end
    try switches.debug_degree_optim = retrieve(conf, "debug","degree_optim", Bool) catch KeyError end
    try switches.debug_bounds = retrieve(conf, "debug", "bounds", Bool) catch KeyError end

    try cfg.tol = retrieve(conf, "main", "tol", Float64) catch KeyError end
    try cfg.max_iter = retrieve(conf, "main", "max_iter", UInt64) catch KeyError end
    try cfg.optim = retrieve(conf, "main", "optim", Bool) catch KeyError end
    try cfg.deg_extra = retrieve(conf, "main", "deg_extra", UInt64) catch KeyError end
    try cfg.deg_max = retrieve(conf, "main", "deg_max", UInt64) catch KeyError end
    try cfg.deg = retrieve(conf, "main", "deg", UInt64) catch KeyError end
    try cfg.lanczos_iter = retrieve(conf, "main", "lanczos_iter", UInt64) catch KeyError end
    try cfg.lanczos_num = retrieve(conf, "main", "lanczos_num", UInt64) catch KeyError end
    try cfg.lanczos_dos_σ = retrieve(conf, "main", "lanczos_dos_sigma", Float64) catch KeyError end
    try cfg.lanczos_dos_σ = retrieve(conf, "main", "lanczos_dos_σ", Float64) catch KeyError end

    return nothing
end

end
