# Input-signal generation for driving the models. The file collects the signal
# functions used as scalar inputs c(t), h(t) of the models, each returned as a
# callable of time:
#   - `signal_random`: a smooth random drive, an Ornstein-Uhlenbeck process passed
#     through a lowpass filter and rescaled to a requested range;
#   - `signal_step`:   a deterministic piecewise-constant step between two values.
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

using Random, Interpolations

# --- Ornstein-Uhlenbeck process sampled on the time grid t (exact update) ---
# The OU process is the mean-reverting stochastic process
#     dX = θ (μ - X) dt + σ dW,
# with reversion rate θ, long-run mean μ and noise intensity σ. It is sampled on
# the given time grid with its exact one-step update, so the statistics do not
# depend on the step size. A first-order (exponential) lowpass filter with time
# constant τ then removes the fastest fluctuations. For the random drive the mean
# is fixed to μ = 0 and the process starts from x0 = μ; the filtered signal is
# finally rescaled so that its realised range matches the user-provided [lo, hi].
# The step signal switches from one constant to another at a given time.
function ornstein_uhlenbeck(t; θ, μ, σ, x0 = μ, rng = Random.default_rng())
    x = similar(collect(t), Float64); x[1] = x0
    for n in 1:length(t) - 1
        a = exp(-θ * (t[n + 1] - t[n]))                       # decay over the step
        x[n + 1] = μ + (x[n] - μ) * a + σ * sqrt((1 - a^2) / (2θ)) * randn(rng)
    end
    return x
end

# --- First-order (exponential) lowpass filter with time constant τ ---
function lowpass(t, x; τ)
    y = similar(x); y[1] = x[1]
    for n in 1:length(t) - 1
        α = 1 - exp(-(t[n + 1] - t[n]) / τ)                   # smoothing weight
        y[n + 1] = y[n] + α * (x[n + 1] - y[n])
    end
    return y
end

# --- Random input signal (OU + lowpass) rescaled to [lo, hi], as a callable of time ---
function signal_random(t; θ, σ, τ, lo, hi, rng = Random.default_rng())
    y = lowpass(t, ornstein_uhlenbeck(t; θ, μ = 0.0, σ, rng); τ)
    ymin, ymax = extrema(y)
    z = lo .+ (hi - lo) .* (y .- ymin) ./ (ymax - ymin)      # rescale to the requested range
    return linear_interpolation(t, z; extrapolation_bc = Flat())
end

# --- Piecewise-constant step: value `a` before `t_switch`, `b` from `t_switch` on ---
function signal_step(a, b, t_switch)
    return t -> t < t_switch ? a : b
end
