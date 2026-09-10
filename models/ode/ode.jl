# Aggregates the ODE models in this directory. The shared `ODEModel` interface is
# included first, then the individual models that each construct one.
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt
include(joinpath(@__DIR__, "ODEModel.jl"))
include(joinpath(@__DIR__, "LorenzSystem.jl"))
include(joinpath(@__DIR__, "RoesslerSystem.jl"))
include(joinpath(@__DIR__, "DuffingOscillator.jl"))
