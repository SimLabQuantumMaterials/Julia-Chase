__precompile__()
module ChASE

include("ChASE/modules.jl")
include("ChASE/Common.jl")
include("ChASE/Filter.jl")
include("ChASE/RayleighRitz.jl")
include("ChASE/Lanczos.jl")

# functions / types that should be present in main module
precond = ChASE_Common.precond # precondition struct
verbosity = ChASE_Common.verbosity # verbosity struct
config = ChASE_Common.config # confituration struct
matrix_input = ChASE_Common.matrix_input # struct for matrix input from file
vector_input = ChASE_Common.vector_input # struct for vector input from file
get_config! = ChASE_Common.get_config! # read config from .ini file
generate_precondition = ChASE_Common.generate_precondition #= generate
    precondition struct from matrix input / vector input or julia matrix /
    vector =#
read_matrix = ChASE_Common.read_matrix # read matrix using matrix_input
read_vector = ChASE_Common.read_vector # read vector using vector_input

# TODO for now only double precision works.
"""
Run the ChASE algorithm. Matrix `A` can be either a julia matrix or a
`matrix_input` object. `nev` and `nex` are required, `cfg::config` and
`log::verbosity` are optional configuration parameters.
Return `Λ, Y, Λ_tilde, V` (see `https://arxiv.org/pdf/1805.10121.pdf`, Alg. 1).
`Λ` and `Y` are `nev` eigenvalues / eigenvectors to the matrix `A`,
`Λ_tilde` and `V` contain `nex` (potentially unconverged) eigenpairs that are
needed for repeated execution if a precondition is defined in `cfg.approx`
example usage:
```
pre = ChASE.generate_precondition( hcat(Y,V), vcat(Λ,Λ_tilde) )
cfg.approx = pre
```

```
function run( A, nev::Int, nex::Int, cfg::config; log = nothing )
function run( A, nev::Int, nex::Int; log = nothing )
```
"""
function run( A, nev::Int, nex::Int, cfg::config; log = nothing )
    # reset counters (important if executed repeatedly)
    ChASE_Lanczos.MATVEC_reset()
    ChASE_RayleighRitz.MATVEC_reset()
    ChASE_Filter.MATVEC_reset()

    # setting the loglevel
    if isnothing(log)
        log = verbosity()
    elseif typeof(log) == verbosity
        # continue
    else
        @warn "log set to unsupported type, using defaults"
        log = verbosity()
    end

    # initialize timer output
    timer = TimerOutput()

    # input handling
    @timeit_debug timer "io" begin
        if typeof(A) <: Hermitian || typeof(A) <: Symmetric
            mat = A
        elseif typeof(A) == matrix_input
            mat = ChASE_Common.read_matrix(A)
        else
            error("A must be either a Hermitian or Symmetric matrix " *
                  "or of the matrix_input type")
        end
    end
    if size(mat,1) != size(mat,2)
        error("A must be square")
    end
    if size(mat,1) < 500
        @warn "For the ChASE solver to work properly, the dimension should be >= 500"
    end
    approx::Bool = false
    if isnothing(cfg.approx)
        @info "using solver without values for V and λ"
    elseif typeof(cfg.approx) == precond
        approx = true
        @info "using solver with approximate values for V and λ"
    else
        @warn "wrong type of argument given for cfg.approx. continuing without"
    end
    nev = UInt64(nev)
    nex = UInt64(nex)
    N::UInt64 = size(mat,1)
    if nev <= 0.5*N && nev >= 0.2*N
        @warn "nev requested between 0.2N and 0.5N"
    elseif nev > 0.5*N
        error("not a reasonable value for nev (>0.5N)")
    end
    if nex > max(15,0.1*N)
        @warn "nex should be smaller than max(15,0.1N)"
    end

    # initialize eigenvectors / eigenvalues with `nothing`, these variables are
    # set conditionally depending on `approx`
    V = nothing
    λ = nothing
    Λ_tilde = nothing
    if approx
        if size(cfg.approx.V) != (N,nev+nex)
            error("size(V) must be (N,nev+nex)")
        end
        if size(cfg.approx.λ) != (nev+nex,)
            error("size(λ) must be (nev+nex,)")
        end
        if cfg.approx.λ[1] > cfg.approx.λ[end]
            error("λ[1] larger than λ[end]")
        end
        V = cfg.approx.V
        λ = cfg.approx.λ
    end

    # more checks on the input configuration
    if cfg.tol <= 1.e-15
        @warn "tolerance below 1.e-15"
    end
    if cfg.max_iter > 1000
        @warn "max_iter above 1000"
    end
    if cfg.deg >= cfg.deg_max
        @warn "degree bigger than max degree"
    end
    if cfg.optim
        if cfg.deg >= 12
            @warn "with optim=true, deg should be <12"
        end
    end
    if cfg.deg_max > 50
        @warn "max degree too high (>50)"
    end
    if cfg.deg_extra > 8
        @warn "extra degree too high (>8)"
    end
    if cfg.lanczos_iter > 100
        @warn "more than 100 lanczos iterations"
    end
    if cfg.lanczos_num > 25
        @warn "more than 25 lanczos vectors"
    end

    # initialize degrees array
    m = Array{UInt64,1}(undef,nev+nex)
    for i=1:nev+nex
        m[i] = cfg.deg
    end

    # complex or real
    ctype = ChASE_Common.choose_complex_type( mat )

    n_found::Int = 0 # TODO for some reason, one is not allowed to declare n_found inside the macro
    @timeit_debug timer "total" begin
    # initialize output variables
    Y = Array{ctype,2}(undef,N,nev)
    Λ = Array{Float64,1}(undef,nev)

    ### LANCZOS ###
    @timeit_debug timer "lanczos" begin
        if isnothing(V) && isnothing(λ)
            μ_1, μ_nevnex, b_sup, V = ChASE_Lanczos.lanczos(mat, cfg.lanczos_iter,
                                           cfg.lanczos_num, nev, nex;
                                           approx = false, σ_gauss = cfg.lanczos_dos_σ )
        else
            b_sup = ChASE_Lanczos.lanczos( mat, cfg.lanczos_iter, cfg.lanczos_num,
                                           nev, nex; approx = true,
                                           σ_gauss = cfg.lanczos_dos_σ )
            μ_1 = λ[1]
            μ_nevnex = λ[end]
        end
    end

    n_found_per_step = Array{Int}(undef,0)

    while_counter = 0
    # main loop
    while n_found < nev

        # counter for max iter
        if while_counter > cfg.max_iter
            @warn "reached max_iter. breaking loop"
            break
        end

        ### FILTER ###
        @timeit_debug timer "filter" begin
            V = ChASE_Filter.filter!( mat, V, μ_1, μ_nevnex, b_sup, m )
        end
        ### QR ###
        @timeit_debug timer "qr" begin
            YV_qr = hcat(Y[:,1:n_found],V)
            Q = ChASE_Common.fast_qr!( YV_qr )[:,n_found+1:nev+nex]
        end
        ### RAYLEIGH RITZ ###
        @timeit_debug timer "rr" begin
            V, Λ_tilde = ChASE_RayleighRitz.solve( mat, Q )
        end
        ### RESIDUALS ###
        @timeit_debug timer "residuals" begin
            Res = ChASE_RayleighRitz.residuals( mat, V, Λ_tilde )
        end

        ### DEFLATION & LOCKING ###
        n_found_const = n_found
        n_found_current::Int = 0
        for a=1:nev-n_found_const
            if Res[a] > cfg.tol
                break
            end
            Λ[n_found+1] = Λ_tilde[a]
            Y[:,n_found+1] = V[:,a]
            n_found += 1
            n_found_current += 1
        end
        V = V[:,n_found_current+1:end]
        Res = Res[n_found_current+1:end]
        Λ_tilde = Λ_tilde[n_found_current+1:end]

        push!(n_found_per_step,n_found) # save how many values were found / step

        ### DEGREES ###
        μ_1 = minimum(vcat(Λ[1:n_found],Λ_tilde))
        μ_nevnex = maximum(vcat(Λ[1:n_found],Λ_tilde))
        if cfg.optim
            c = 0.5 * (b_sup + μ_nevnex)
            e = 0.5 * (b_sup - μ_nevnex)
            for a=1:nev+nex-n_found
                m[a] = ChASE_Filter.degrees( cfg.tol, Res[a], Λ_tilde[a], c, e;
                                             deg_extra = cfg.deg_extra,
                                             deg_max = cfg.deg_max )
            end
            resize!(m,nev+nex-n_found)

            # sorting according to degrees
            indices = sortperm(m)
            Res = Res[indices]
            m = m[indices]
            tmpvec = Array{ChASE_Common.choose_complex_type(mat),1}(undef,N)
            for i=1:size(m,1)
                ChASE_Common.swap_col!(V,tmpvec,i,indices[i])
            end
            Λ_tilde = Λ_tilde[indices]
        else
            resize!(m,nev+nex-n_found)
        end

        # debug information
        while_counter += 1
        if log.debug_residuals
            @info "dbg-residuals" Res
        end
        if log.debug_eigenvalues
            @info "dbg-eigenvalues" Λ_tilde
        end
        if log.debug_degree_optim
            @info "dbg-degrees" Array{Int,1}(m)
        end
        if log.debug_bounds
            @info "dbg-bounds" μ_1 μ_nevnex b_sup n_found
        end
    end # while

    end # total timer

    # timer output
    @info "Timers" timer
    # counter outputs
    if log.counter_matvec
        MATVEC_counter = ChASE_Lanczos.MATVEC_counter +
                        ChASE_RayleighRitz.MATVEC_counter +
                        ChASE_Filter.MATVEC_counter
        @info "MATVEC counter" MATVEC_counter ChASE_Lanczos.MATVEC_counter ChASE_RayleighRitz.MATVEC_counter ChASE_Filter.MATVEC_counter
    end
    if log.counter_while
        @info "while counter" while_counter
    end
    if log.counter_per_step
        @info "found per step" n_found_per_step
    end


    # returns both nev and nex
    return Λ, Y, Λ_tilde, V
end

# overload with default config
function run( A, nev::Int, nex::Int; log = nothing )
    cfg = config()
    return run( A, nev, nex, cfg; log = log )
end

end

"""
`enable` fine-grained timers for each part of the algorithm

```
function ChASE_timers( enable::Bool )
```
"""
function ChASE_timers( enable::Bool )
    if enable
        ChASE.TimerOutputs.enable_debug_timings(ChASE)
    else
        ChASE.TimerOutputs.disable_debug_timings(ChASE)
    end
end

nothing
