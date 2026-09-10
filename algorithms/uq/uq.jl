# Aggregates the uncertainty-quantification algorithms in this directory: a generic
# rollout-error scorer usable with any model exposing the `predict` interface, and a
# Gaussian-process surrogate that turns those scattered error samples into a continuous
# mean/variance "accuracy map" over state space.
#
# ----------------------------------------------------------------------------------
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt
include(joinpath(@__DIR__, "RolloutError.jl"))
include(joinpath(@__DIR__, "GPErrorModel.jl"))
