<!-- Copyright (c) 2026 Dirk Hartmann, TU Darmstadt -->
# Student Exercises

This folder contains student exercises that build on top of the algorithms and models documented in the main documentation. Each exercise combines existing algorithms and scripts in new ways, featuring comparisons, parameter sweeps, refinements, and other explorations.

## Exercise Sets

| Topic | File | Description |
| --- | --- | --- |
| **Machine Learning** | [ml_exercises.md](ml_exercises.md) | Architecture sweeps, GPU timing, noise robustness, optimizer comparisons, finite-difference vs. neural-ODE refinements for Lorenz, Duffing, and Rössler predictors |
| **Reduced-Order Models** | [rom_exercises.md](rom_exercises.md) | Comparing POD, Krylov, and neural reduction methods on PCB and SLM problems; autoregressive refinement studies |
| **Uncertainty Quantification** | [uq_exercises.md](uq_exercises.md) | Gaussian-process error estimators for reduced-order models of driven systems |

## Structure

- **`*_exercises.md` files** (in this folder) — contain **task descriptions only** for students
- **`ml/ml_exercises.md`, `rom/rom_exercises.md`, `uq/uq_exercises.md`** — contain full answers, results, and discussion (instructor materials, not committed to git)
- **`ml/results/`, `rom/results/`, `uq/results/`** — generated outputs from running exercise scripts (not committed to git)
- **`ml/*.jl`, `rom/*.jl`, `uq/*.jl`** — exercise implementation scripts (committed to git)

## Agentic Software Development
To support agentic software development, for example using GitHub Copilot ([GitHub Education](https://github.com/education/students)), corresponding project guidelines ([`copilot-instructions.md`](.github/copilot-instructions.md)) are included. I recommend trying Copilot to explore the provided algorithms through the exercises or your own examples. Though, please keep in mind that this does not replace the responsibility of understanding the algorithms in detail, [for the greater good](https://www.nature.com/articles/s44271-026-00402-1).