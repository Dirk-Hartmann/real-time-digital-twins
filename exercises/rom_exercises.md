# Reduced-Order-Model Exercises — Comparing Reductions of the PCB and SLM Heat Problems


## Contents

- [Exercise 1 — Comparing four reductions at the same dimension](#exercise-1--comparing-four-reductions-at-the-same-dimension)
- [Exercise 2 — Autoregressive refinement of the operator-inference model](#exercise-2--autoregressive-refinement-of-the-operator-inference-model)
- [Exercise 3 — A neural network instead of a polynomial reduced model](#exercise-3--a-neural-network-instead-of-a-polynomial-reduced-model)
- [Exercise 4 — Accuracy vs. latent dimension for the SLM problem](#exercise-4--accuracy-vs-latent-dimension-for-the-slm-problem)

## Exercises

### Exercise 1 — Comparing four reductions at the same dimension

[KrylovROM.jl](../algorithms/rom/KrylovROM.jl),
[POD-Reduction.jl](../algorithms/rom/POD-Reduction.jl),
[AE-POD-Reduction.jl](../algorithms/rom/AE-POD-Reduction.jl) and
[AE-CNN-Reduction.jl](../algorithms/rom/AE-CNN-Reduction.jl) each reduce the PCB
temperature field a different way. Build all four at the **same** reduced dimension
$r = \ell = 4$ (training the two autoencoders with a 4-dimensional latent space) and

1. visually compare their four dominant modes, for Krylov/POD the basis columns
   $V[:,j]$, for the autoencoders the field obtained by decoding the $j$-th latent
   unit vector (relative to decoding the origin);
2. compare their accuracy by the mean absolute error (MAE) of the **projection**
   (encode a validation snapshot to the reduced/latent space and decode it straight
   back - no time stepping) on a held-out validation set.

Split the 10 trajectories of `data/PCB_training.bson` **by trajectory**, 80% (8) for
training and 20% (2) for validation, so no snapshot of a validation trajectory leaks
into training (unlike the pooled-column 80/20 split used by
[PCB_POD.jl](../scripts/rom/PCB_POD.jl),
[PCB_AutoencoderPOD.jl](../scripts/rom/PCB_AutoencoderPOD.jl) and
[PCB_AutoencoderCNN.jl](../scripts/rom/PCB_AutoencoderCNN.jl)).


### Exercise 2 — Autoregressive refinement of the operator-inference model

[PCB_OperatorInference.jl](../scripts/rom/PCB_OperatorInference.jl) fits the
reduced polynomial operators $(A, C)$ of the PCB heat problem by one-shot **least
squares** on finite-difference derivative labels. Because the fitted weights $\theta$
are decoupled from the training strategy, they can be handed to
`train_model_autoregressive` ([AutoregressiveTraining.jl](../algorithms/ml/AutoregressiveTraining.jl))
as a starting point and refined further with the **autoregressive** (unrolled Euler
rollout) objective,  mirroring the two-stage recipe of
[Lorenz_poly_finitediff_autoregressive_refinement.jl](../exercises/ml/Lorenz_poly_finitediff_autoregressive_refinement.jl).
Unlike the autonomous Lorenz exercises, the PCB system is **driven** by the recorded
signal $u=[c;h]$, so `rollout_loss`/`train_model_autoregressive` were extended with an
optional per-step control argument `Useg` (defaulting to zero, so every existing caller
is unaffected). Compare the least-squares-only model against the refined one, driven by
the same recorded signal, against the full-order reference and the POD projection floor.


### Exercise 3 — A neural network instead of a polynomial reduced model

Repeat Exercise 2's two-stage recipe, but replace the polynomial reduced
model with a **neural network** vector field (`neural_model`, NeuralModel.jl) acting on
the same $r=8$ POD coordinates. Since a network is not linear in its weights, stage 1
uses `train_model_residual` (pointwise finite-difference regression) instead of the
one-shot least-squares solve; stage 2 continues training with
`train_model_autoregressive`, as before. Compare the finite-difference-only fit against
the refined one, and against the polynomial results of Exercise 2.


### Exercise 4 — Accuracy vs. latent dimension for the SLM problem

Repeat the reduction comparison of Exercise 1, but this time for the moving
laser-spot SLM problem (`data/SLM_training.bson`) and as a **sweep over reduced
dimension** rather than a single shared size. For the linear POD basis
([POD-Reduction.jl](../algorithms/rom/POD-Reduction.jl)), score every mode count
$r=1,\dots,100$ by literally projecting a snapshot back and forth (encode then
decode). For the dense POD-autoencoder
([AE-POD-Reduction.jl](../algorithms/rom/AE-POD-Reduction.jl), $p=100$ POD
pre-projection) and the convolutional autoencoder
([AE-CNN-Reduction.jl](../algorithms/rom/AE-CNN-Reduction.jl)), retrain from scratch at
latent sizes $\ell\in\{4,8,16,32,64\}$. Split the 10 trajectories 80/20 **by
trajectory** (as in Exercise 1) and plot the relative reconstruction error against the
reduced dimension for both the training and validation sets.

