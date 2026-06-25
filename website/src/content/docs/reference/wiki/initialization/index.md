---
title: "Initialization"
description: "All 14 public types in the AiDotNet.initialization namespace, organized by kind."
section: "API Reference"
---

**14** public types in this namespace, organized by kind.

## Models & Types (8)

| Type | Summary |
|:-----|:--------|
| [`EagerInitializationStrategy<T>`](/docs/reference/wiki/initialization/eagerinitializationstrategy/) | Eager initialization strategy that initializes weights immediately on construction. |
| [`FromFileInitializationStrategy<T>`](/docs/reference/wiki/initialization/fromfileinitializationstrategy/) | Initialization strategy that loads weights from an external file. |
| [`HeInitializationStrategy<T>`](/docs/reference/wiki/initialization/heinitializationstrategy/) | He/Kaiming initialization strategy for ReLU-family activations. |
| [`LazyInitializationStrategy<T>`](/docs/reference/wiki/initialization/lazyinitializationstrategy/) | Lazy initialization strategy that defers weight allocation until first Forward() call. |
| [`LeCunInitializationStrategy<T>`](/docs/reference/wiki/initialization/lecuninitializationstrategy/) | LeCun initialization strategy for SELU activations and self-normalizing networks. |
| [`OrthogonalInitializationStrategy<T>`](/docs/reference/wiki/initialization/orthogonalinitializationstrategy/) | Orthogonal initialization strategy for RNNs, LSTMs, and deep networks. |
| [`UniformInitializationStrategy<T>`](/docs/reference/wiki/initialization/uniforminitializationstrategy/) | Uniform random initialization with configurable range. |
| [`ZeroInitializationStrategy<T>`](/docs/reference/wiki/initialization/zeroinitializationstrategy/) | Zero initialization strategy that sets all weights to zero. |

## Base Classes (1)

| Type | Summary |
|:-----|:--------|
| [`InitializationStrategyBase<T>`](/docs/reference/wiki/initialization/initializationstrategybase/) | Base class for initialization strategies providing common functionality. |

## Interfaces (1)

| Type | Summary |
|:-----|:--------|
| [`IInitializationStrategy<T>`](/docs/reference/wiki/initialization/iinitializationstrategy/) | Defines a strategy for initializing neural network layer parameters. |

## Enums (2)

| Type | Summary |
|:-----|:--------|
| [`InitializationStrategyType`](/docs/reference/wiki/initialization/initializationstrategytype/) | Specifies the type of initialization strategy to use for layer weights. |
| [`WeightFileFormat`](/docs/reference/wiki/initialization/weightfileformat/) | Specifies the format of a weight file. |

## Helpers & Utilities (2)

| Type | Summary |
|:-----|:--------|
| [`InitializationStrategies<T>`](/docs/reference/wiki/initialization/initializationstrategies/) | Provides factory methods and default instances for initialization strategies. |
| [`InitializationStrategy<T>`](/docs/reference/wiki/initialization/initializationstrategy/) | Provides backward-compatible access to initialization strategies. |

