# Project Guidelines

This repository is a curated collection of compact, educational **Julia** algorithms that illustrate the core numerical methods enabling real-time digital twins. It supports a course whose goal is to make students familiar with these algorithms through small, self-contained, runnable examples. Every algorithm should be underpinned by a clear worked example.

## Language and Style

- Implement all algorithms in **Julia**.
- Optimize for **compactness and expressiveness over efficiency or execution speed**. Prefer short, vectorized, idiomatic Julia (broadcasting, comprehensions, `\` for linear solves) to long imperative loops, unless clarity demands otherwise.
- Favor readability for classroom use: an algorithm should read like the math it implements.
- Add comments where they aid understanding, and **separate distinct tasks into clearly marked sections** with a short header comment (e.g. `# --- Assemble conductivity matrix ---`, `# --- Solve ---`, `# --- Update ---`). Keep the number of sections small and meaningful.
- **Keep in-code comments minimal.** Put longer explanations (method, derivations, rationale) in the top-of-file header comment block, and keep inline comments to concise section headers.
- Keep examples self-contained and minimal so they can be read top-to-bottom in one sitting.
- **Naming:** name algorithm files in `CamelCase` after their primary concept (e.g. `FiniteElement.jl`, `TopologyOptimization.jl`). Define **plain functions** in these files—do not wrap algorithms in a `module`. Use `snake_case` for function and variable names.
- **Type association:** express a function's link to a type through first-argument dispatch (`node_id(g::Grid, …)`), not name prefixes like `Grid_node_id`. Add short docstrings tying accessors to their type.

## Dependencies

- **Use existing libraries for standard subtasks** — e.g. solving linear systems or training ML models — unless the task is specifically to implement that primitive. Time-stepping schemes (implicit Euler, linearly-implicit trapezoidal, …) are themselves the algorithms this repo teaches, so they are implemented by hand rather than delegated to an ODE-solver package.
- **Keep the external dependency footprint minimal.** Prefer a few broad, high-coverage packages over many narrow ones. Reach first for: `Optimization` (with `OptimizationOptimisers` for Adam and `OptimizationOptimJL` for LBFGS/BFGS) plus `Zygote` for gradient-based training, `Lux`/`ComponentArrays` for neural-network models (`LuxCUDA` when a script needs the GPU), and `BSON` for persistence. Only add a new dependency when none of the established ones fit — e.g. `GaussianProcesses.jl` (uncertainty quantification) and `Images.jl` (PNG-based material/source fields) were added because no established package covered that capability.
- Persist generated data as **`BSON`** and produce all plots with **`Plots.jl`**.

## Reuse and Consistency

- **Reuse previously built algorithms** rather than reimplementing them. For example, use an existing simulation algorithm to generate training data for a data-driven algorithm.
- **Be consistent across algorithms**: shared naming conventions, function signatures, section structure, data formats, and utility usage. New algorithms should look and feel like existing ones in this repository.
- When there is reasonable doubt or more than one viable alternative, seek user feedback before proceeding.
- Place broadly useful helpers in `util/` and reuse them instead of duplicating code.

## Repository Structure

- `util/` — reusable utility algorithms shared across the collection.
- `models/` — **physical model examples** (organised by domain, e.g. `models/pde/`) used throughout the lecture. Keep these focused on the model/discretisation itself; no plotting. Static input assets (e.g. geometry PNGs) live in `models/data/`.
- `algorithms/` — the **core algorithms**. Keep these focused on the method itself; no plotting or visualization here.
- `data/` — generated datasets, persisted as `BSON` by the scripts that create them. Nothing under it is synchronized with the repository except the folder's `README.md`; every dataset is regenerated on the fly by rerunning the producing script.
- `results/` — **generated outputs** (figures, persisted solutions) written by `scripts/`, organised by domain (e.g. `results/pde/`). Unlike `data/`, this folder IS tracked by git, so avoid leaving stray/superseded files behind when a script's output changes.
- `scripts/` — scripts that run the algorithms and produce **all visualization/plots** with `Plots.jl`.
- `exercises/` — **student exercises** that combine existing algorithms/scripts in a new way (comparisons, sweeps, refinements, …), organised by domain (e.g. `exercises/rom/`) with their own `results/` subfolder; documented in `exercises/<domain>_exercises.md`.
- `tests/` — tests that validate the algorithms.

## Conventions

- Core algorithms in `algorithms/` are pure and visualization-free; do all plotting in `scripts/`.
- In scripts, visualization and figure creation must always be at the very end of the file, separated by `# ============================== VISUALIZATION ===============================`.
- **Progress tracking:** simulation and training loops should include regular progress updates via `println` to keep the user informed of execution status (e.g., `i % 2000 == 0 && println("  step $i / $(length(ts))")`).
- When an algorithm consumes data, prefer loading it from `data/` (produced by another algorithm in this repo) over generating it inline.
- Provide or update a runnable example in `scripts/` and a validation in `tests/` for each new algorithm.
- **Organise by domain subfolder** (`models/pde/`, `algorithms/rom/`, `scripts/pde/`, `exercises/rom/`, `tests/pde/`). Each code subdirectory (`util/`, `models/<domain>/`, `algorithms/<domain>/`) has an aggregator file named after the folder (`util/util.jl`, `models/pde/pde.jl`) that `include`s its siblings; downstream files include the aggregator. `scripts/`, `exercises/`, and `tests/` are consumers, not libraries, so their domain subfolders have no aggregator of their own. Scripts and tests resolve  repo-relative includes with a small self-contained helper instead of repeated `joinpath`:
  ```julia
  const ROOT = normpath(@__DIR__, "..", "..")
  use(parts...) = Base.include(Main, joinpath(ROOT, parts...))
  ```

- **Train/validation splits must never mix trajectories.** When a script or exercise holds out data for validation (or test), split on whole trajectories — e.g. with [`split_trajectories`](../util/DataSplit.jl) — and index the trajectory dimension (e.g. `X[:, :, train]`), never shuffle/split individual snapshots pooled across trajectories. Every snapshot of a given trajectory must fall entirely in one split; no trajectory may contribute to both.

## Execution

- **Always ask before executing.** Do not run scripts, tests, benchmarks, or any Julia code (including package installs) until the user has reviewed the code and explicitly approved. Prepare and edit the code first, then wait for the go-ahead before running anything.

## Documentation

- Document each domain in `docs/<domain>.md` (e.g. `docs/pde.md`); a domain with exercises also gets an `exercises/<domain>_exercises.md` (e.g. `exercises/rom_exercises.md`) stating each exercise's task, the script that answers it, and the results obtained. Keep shared symbols in `docs/notation.md`, and index every doc in `docs/README.md` (and in the root `README.md`'s shorter summary list).
- Keep in-code comments minimal but name sections consistently; put derivations and ASCII-art numbering diagrams in the docs. Update the relevant doc — including its repository-layout table and any prose describing a script's inputs/outputs — whenever code is added, renamed, or its behaviour changes (e.g. whether it persists a `BSON` file).

## Notation

Follow `docs/notation.md`. In particular:

- Unless otherwise stated, denote the state of a dynamical system by `X`, respectively `x`, and any external (control) input by `U`, respectively `u`.
- Matrices and linear operators use **uppercase** (`M`, `K`); load vectors and scalars use  **lowercase** (`s_c`, `s_h`, `c`, `h`); the primary nodal unknown `T` stays uppercase.
- Spatial step is `δx` (`δy`), temporal step `δt`.
- Cell-wise input fields use a `field_` prefix (`field_κ`, `field_s_c`, `field_s_h`).
- Unicode/Greek identifiers (`κ`, `δx`) are welcome.
