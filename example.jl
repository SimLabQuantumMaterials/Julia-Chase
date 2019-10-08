#!/usr/bin/julia
include("ChASE.jl")

ChASE_timers(true)
verb = ChASE.verbosity()
verb.debug_bounds = true
cfg = ChASE.config()
cfg.optim = true
cfg.lanczos_iter = 100

A = ChASE.matrix_input(filename="NaCl_3893_1-1.bin", matrix_size=3893, filetype="bin")

# config argument optional, fall back to defaults if nothing given
λ, V, λ_tilde, V_tilde = ChASE.run( A, 256, 40, cfg; log = verb )

@info "" λ λ_tilde

