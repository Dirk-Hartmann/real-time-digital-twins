# Aggregates the reduced-order model algorithms in this directory. Two shared
# interfaces come first — `ReducedModel` (encode/decode/predict) and the reduction-
# only `DimReduction` (encode/decode, no prediction) — then the algorithms that each
# construct one: the projection ROMs (Krylov, POD + Galerkin) and the dimensionality
# reductions (linear POD, learned autoencoder). Requires the PDE models for the full
# operators and the util grid.
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt
include(joinpath(@__DIR__, "ReducedModel.jl"))
include(joinpath(@__DIR__, "DimReduction.jl"))
include(joinpath(@__DIR__, "KrylovROM.jl"))
include(joinpath(@__DIR__, "PODGalerkinROM.jl"))
include(joinpath(@__DIR__, "POD-Reduction.jl"))
include(joinpath(@__DIR__, "AE-POD-Reduction.jl"))
include(joinpath(@__DIR__, "AE-CNN-Reduction.jl"))
