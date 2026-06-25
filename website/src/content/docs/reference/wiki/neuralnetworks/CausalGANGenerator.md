---
title: "CausalGANGenerator<T>"
description: "Causal-GAN generator that learns causal graph structure (directed acyclic graph) and generates synthetic data respecting causal relationships between features."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.SyntheticData`

Causal-GAN generator that learns causal graph structure (directed acyclic graph)
and generates synthetic data respecting causal relationships between features.

## For Beginners

Causal-GAN learns which features cause other features:

Instead of just learning that "Age and Income are related" (correlation),
it learns "Education causes higher Income" (causation).

This has two benefits:

1. Generated data respects cause-effect chains, producing more realistic samples
2. You can simulate "what-if" scenarios (interventions) on specific features

The model learns a weighted adjacency matrix W where W[i,j] means feature i
influences feature j. A DAG penalty (NOTEARS) ensures no circular dependencies.

The training uses:

- WGAN-GP loss for stable adversarial training
- NOTEARS penalty via augmented Lagrangian for DAG acyclicity constraint
- L1 sparsity on the adjacency matrix for interpretable causal structure

The NOTEARS constraint is: h(W) = tr(e^(W * W)) - d = 0
where d is the number of features. This is zero if and only if W encodes a DAG.
The matrix exponential is computed via truncated Taylor series.

This implementation follows the standard neural network architecture pattern with:

- Proper inheritance from NeuralNetworkBase
- Layer-based architecture using ILayer components
- Engine-based tensor operations for CPU/GPU acceleration
- Full autodiff and JIT compilation support

If you provide custom layers in the architecture, those will be used directly
for the generator network. Otherwise, the network creates the standard architecture.

Example usage:

## How It Works

Causal-GAN discovers causal structure using a NOTEARS-style continuous relaxation
for learning a directed acyclic graph (DAG), then generates each feature as a
function of its causal parents via structural equation models.

Architecture:

Reference: Zheng et al., "DAGs with NO TEARS: Continuous Optimization for Structure Learning" (NeurIPS 2018)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CausalGANGenerator` | Initializes a new instance of the `CausalGANGenerator` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Columns` |  |
| `IsFitted` |  |
| `Options` | Gets the CausalGAN-specific options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyCausalStructure(Tensor<>)` | Applies the learned causal structure: y = (I + W^T) * x. |
| `ApplyCausalStructureBatched(Tensor<>)` | Applies the learned causal adjacency matrix to a batch of fake samples (Kocaoglu et al. |
| `ApplyOutputActivations(Tensor<>)` | Applies output activations per column type: tanh for continuous, softmax for categorical. |
| `ApplyOutputActivationsBatched(Tensor<>)` | Batched per-column VGM output activation (matches CTGAN / TableGAN's tape-tracked Tanh + per-block Softmax dispatch). |
| `BuildDefaultGeneratorLayers(Int32,Int32)` | Builds default generator layers with residual connections, BatchNorm, and manual ReLU. |
| `BuildDiscriminator(Int32)` | Builds the discriminator network with Dropout and manual LeakyReLU. |
| `ComputeNOTEARSConstraintAndGradient` | Computes the NOTEARS acyclicity constraint and its gradient. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `DiscriminatorForward(Tensor<>,Boolean)` | Discriminator forward pass with pre-activation caching and optional Dropout. |
| `DiscriminatorForwardBatched(Tensor<>,Boolean)` | Batched discriminator forward (LeakyReLU 0.2 + dropout), tape-tracked. |
| `Fit(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32)` | Fits the CausalGAN generator to the provided real tabular data. |
| `FitAsync(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32,CancellationToken)` |  |
| `Generate(Int32,Vector<>,Vector<>)` | Generates new synthetic tabular data rows respecting the learned causal structure. |
| `GeneratorForward(Tensor<>)` | Generator forward pass with residual connections and pre-activation caching. |
| `GeneratorForwardBatched(Tensor<>)` | Batched generator forward, tape-tracked. |
| `GetFeatureImportance` |  |
| `GetModelMetadata` |  |
| `InitializeLayers` | Initializes the layers of the CausalGAN generator based on the provided architecture. |
| `PredictCore(Tensor<>)` | Runs the generator forward pass with residual connections and pre-activation caching. |
| `RebuildLayersWithActualDimensions(Int32,Int32,Int32)` | Rebuilds generator and discriminator layers with actual data dimensions discovered during Fit(). |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `TrainDiscriminatorStepBatched(Matrix<>,Int32)` | Paper-faithful WGAN-GP critic step over the causal-structured fake distribution (Kocaoglu et al. |
| `TrainGeneratorStepBatched(Int32)` | Paper-faithful generator step: minimize -E[D(G(z) ⊗ A)] where the fake samples flow through the residual generator then the causal adjacency mask (Kocaoglu et al. |
| `UpdateAdjacencyAugmentedLagrangian()` | Updates the adjacency matrix using the augmented Lagrangian method for the NOTEARS constraint. |
| `UpdateParameters(Vector<>)` |  |

