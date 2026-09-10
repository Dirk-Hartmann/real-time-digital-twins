# Aggregates the PDE models in this directory (requires the util grid first).
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt
include(joinpath(@__DIR__, "PDEModel.jl"))
include(joinpath(@__DIR__, "DirichletBC.jl"))
include(joinpath(@__DIR__, "thermal_FEM.jl"))
include(joinpath(@__DIR__, "thermal_FVM.jl"))
include(joinpath(@__DIR__, "PCB_thermal.jl"))
include(joinpath(@__DIR__, "SLM_thermal.jl"))
