# Neural-network vector field for data-driven prediction of a dynamical system.
# A small multilayer perceptron models the right-hand side of a controlled ODE,
#             dx/dt ≈ f_θ(x, u),
# with weights θ learned from sampled trajectories. The control u enters as an extra
# network input rather than an additive forcing term, so the network can learn a
# nonlinear response to it. Marching f_θ with one explicit Euler step gives the predictor
#             predict(θ, x, u, δt) = x + δt f_θ(x, u),
# so a `NeuralModel` mirrors the `ODEModel` / `ReducedModel` interface: it bundles
# the weights θ with `rhs(θ, x, u)` and `predict(θ, x, u, δt)` maps that take θ
# explicitly as their first argument. This file holds only the model definition and its
# construction; the weights are fitted by the training strategies in ResidualTraining.jl
# and AutoregressiveTraining.jl, and the model is marched with `predict`.
#
# The network sees the standardised state x̂ = (x - μ) ./ σ stacked with the control u and
# returns the standardised derivative dx̂/dt, so `rhs` scales the output back by σ. The
# control is appended in physical units (no control statistics are assumed). For a batch
# (x̂ a matrix), the control may be a single vector — broadcast across every batch entry,
# for a control shared by the whole batch — or a matrix with one column per batch entry,
# for a control that genuinely varies per sample (e.g. a recorded time-varying drive). The
# fixed statistics μ, σ (per component, from the training data) are folded into the maps as
# constants, not trainable weights, which keeps the state-space interface in physical
# units while giving the tanh layers well-scaled state inputs.
#
# ----------------------------------------------------------------------------------
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

using Lux, ComponentArrays, Random

struct NeuralModel
    params    # ComponentArray of network weights θ
    rhs       # (θ, x, u)      -> f_θ(x, u)             standardised state / control in
    predict   # (θ, x, u, δt)  -> x + δt f_θ(x, u)      one explicit Euler step
end

# --- Stack the standardised state x̂ over the control u to form the network input: a single
#     control vector is broadcast across the batch, while a control matrix (one column per
#     batch entry — e.g. a time-varying drive sampled once per snapshot) is used as-is ---
stack_input(x̂::AbstractVector, u) = vcat(x̂, u)
stack_input(x̂::AbstractMatrix, u::AbstractVector) = vcat(x̂, repeat(u, 1, size(x̂, 2)))
stack_input(x̂::AbstractMatrix, u::AbstractMatrix) = vcat(x̂, u)

# --- Build an MLP vector field wrapped in the model interface (μ, σ fold in state scaling) ---
function neural_model(; nin = 3, nctrl = nin, width = 32, depth = 2, act = tanh,
                        μ = zeros(nin), σ = ones(nin), rng = Random.default_rng())
    hidden = ntuple(_ -> Dense(width => width, act), depth - 1)
    net = Chain(Dense(nin + nctrl => width, act), hidden..., Dense(width => nin))
    ps, st = Lux.setup(rng, net)
    params = ComponentArray(ps)
    rhs(θ, x, u)      = σ .* net(stack_input((x .- μ) ./ σ, u), θ, st)[1]
    predict(θ, x, u, δt) = x .+ δt .* rhs(θ, x, u)
    return NeuralModel(params, rhs, predict)
end
