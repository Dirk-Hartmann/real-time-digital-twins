# Aggregates the machine-learning algorithms in this directory. The learned vector-field
# models are included first — a neural-network model and a quadratic polynomial model —
# then the training strategies that fit their parameters: residual (derivative-matching),
# Sobolev (derivative-and-Jacobian-matching), autoregressive (unrolled neural-ODE), and
# one-shot least-squares (operator inference) training. Each learned model is marched
# with explicit Euler and shares the model interface used by the ODE and reduced-order
# models, so the models are fitted by the same training strategies.
#
# ----------------------------------------------------------------------------------
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt
include(joinpath(@__DIR__, "NeuralModel.jl"))
include(joinpath(@__DIR__, "PolynomialModel.jl"))
include(joinpath(@__DIR__, "ResidualTraining.jl"))
include(joinpath(@__DIR__, "SobolevTraining.jl"))
include(joinpath(@__DIR__, "AutoregressiveTraining.jl"))
include(joinpath(@__DIR__, "LeastSquaresTraining.jl"))
