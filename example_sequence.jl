#!/usr/bin/julia
include("ChASE.jl")

cfg = ChASE.config()
verb = ChASE.verbosity()
verb.counter_per_step = true
cfg.optim = true
ChASE_timers(true)

sequence = ["NaCl_3893/$(i).bin" for i=2:16]

nev = 256
nex = 40
A = ChASE.matrix_input(filename="NaCl_3893_1-1.bin", matrix_size=3893, filetype="bin")

# solve first iteration without previous results
λ, V, λ_tilde, V_tilde = ChASE.run( A, nev, nex, cfg; log=verb )

seq_ind = 1
for fname in sequence
    global V
    global λ
    global V_tilde
    global λ_tilde
    global cfg
    global verb
    global seq_ind
    seq_ind += 1
    @info "sequence index" seq_ind
    A = ChASE.matrix_input(filename=fname, matrix_size=3893, filetype="bin")
    cfg.approx = ChASE.generate_precondition(hcat(V,V_tilde),
                                             vcat(λ,λ_tilde); checks_tolerance = 1.e-5 )
    λ, V, λ_tilde, V_tilde = ChASE.run( A, nev, nex, cfg; log=verb )
end

