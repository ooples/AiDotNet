---
title: "Autodiff"
description: "All 14 public types in the AiDotNet.autodiff namespace, organized by kind."
section: "API Reference"
---

**14** public types in this namespace, organized by kind.

## Models & Types (4)

| Type | Summary |
|:-----|:--------|
| [`AutogradContext`](/docs/reference/wiki/autodiff/autogradcontext/) | Context object passed to `AutogradFunction` for saving tensors between forward and backward. |
| [`ComparisonResult<T>`](/docs/reference/wiki/autodiff/comparisonresult/) | Result of comparing numerical and analytical gradients. |
| [`TensorOperationsVerification<T>`](/docs/reference/wiki/autodiff/tensoroperationsverification/) | Verifies that TensorOperations autodiff gradients match numerical gradients. |
| [`VerificationSummary<T>`](/docs/reference/wiki/autodiff/verificationsummary/) | Summary of verification results for all operations. |

## Base Classes (1)

| Type | Summary |
|:-----|:--------|
| [`AutogradFunction<T>`](/docs/reference/wiki/autodiff/autogradfunction/) | Base class for custom autograd functions with user-defined forward and backward passes. |

## Options & Configuration (1)

| Type | Summary |
|:-----|:--------|
| [`VerificationConfig<T>`](/docs/reference/wiki/autodiff/verificationconfig/) | Configuration for gradient verification. |

## Helpers & Utilities (8)

| Type | Summary |
|:-----|:--------|
| [`CheckpointingExtensions`](/docs/reference/wiki/autodiff/checkpointingextensions/) | Provides extension methods for gradient checkpointing on computation nodes. |
| [`ComputationNode<T>`](/docs/reference/wiki/autodiff/computationnode/) | Represents a node in the automatic differentiation computation graph. |
| [`Defaults<T>`](/docs/reference/wiki/autodiff/defaults/) | Default configuration for numerical gradient computation. |
| [`GradientCheckpointing<T>`](/docs/reference/wiki/autodiff/gradientcheckpointing/) | Provides gradient checkpointing functionality for memory-efficient training. |
| [`NeuralNetworkDerivatives<T>`](/docs/reference/wiki/autodiff/neuralnetworkderivatives/) | Provides first- and second-order derivatives for neural networks with safe fallbacks. |
| [`NumericalGradient<T>`](/docs/reference/wiki/autodiff/numericalgradient/) | Provides numerical gradient computation using finite differences for gradient verification. |
| [`TensorOperationsVerificationExtensions`](/docs/reference/wiki/autodiff/tensoroperationsverificationextensions/) | Extension methods for TensorOperationsVerification. |
| [`TensorOperations<T>`](/docs/reference/wiki/autodiff/tensoroperations/) | Provides computation graph operations on `ComputationNode` for legacy autodiff. |

