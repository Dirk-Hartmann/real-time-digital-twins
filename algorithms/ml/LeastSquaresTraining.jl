# Least-squares (operator-inference) training for models that are LINEAR in their
# parameters. When the vector field factors as f_θ(x, u) = Θ φ(x, u) with a parameter
# matrix Θ (the stacked coefficient blocks) and a fixed feature vector φ = [x; q(x); u]
# — as for the `PolynomialModel` — matching the derivative estimates d_n reduces to the
# linear least-squares problem
#             min_Θ  Σ_n ‖ Θ φ(x_n, u_n) - d_n ‖²  (+ λ ‖Θ‖²),
# solved in ONE shot (a QR/backslash solve) instead of by iterative gradient descent —
# this is the classic operator-inference fit. The model must expose a `features` map;
# the solved coefficients are packed back into the model's parameter layout. Optional
# Tikhonov regularisation λ stabilises rank-deficient regressors. Unlike the gradient
# strategies there is no loss history: the minimiser is returned directly.
#
# ----------------------------------------------------------------------------------
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

using LinearAlgebra

# --- One-shot least-squares fit of a linear-in-θ model to derivative estimates Y ---
function train_model_leastsquares(model, X, Y; U = zeros(size(X, 1)), λ = 0.0)
    Φ = model.features(X, U)                              # nfeat × N regressor
    if λ > 0
        n = size(Φ, 1)
        Φ = hcat(Φ, sqrt(λ) .* Matrix(I, n, n))
        Y = hcat(Y, zeros(size(Y, 1), n))
    end
    Θ = Y / Φ                                            # min ‖Θ Φ − Y‖  (least-squares solve)

    # --- Pack the coefficient blocks back into the model's parameter layout ---
    θ = copy(model.params); offset = 0
    for k in propertynames(model.params)
        c = size(getproperty(model.params, k), 2)
        getproperty(θ, k) .= Θ[:, offset+1:offset+c]
        offset += c
    end
    return θ
end
