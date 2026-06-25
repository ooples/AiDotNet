---
title: "NeuralProcessBase<T, TInput, TOutput>"
description: "Base class for the Neural Process family of meta-learning algorithms."
section: "API Reference"
---

`Base Classes` · `AiDotNet.MetaLearning.Algorithms`

Base class for the Neural Process family of meta-learning algorithms.
Provides shared infrastructure for encoding context sets, aggregating representations,
and decoding target predictions.

## For Beginners

Neural Processes are like learning a recipe for making predictions:

1. **Context set** = examples you've already seen (like support set in few-shot learning)
2. **Encoder** = summarizes what you've seen into a compact representation
3. **Aggregator** = combines individual summaries into one global summary
4. **Decoder** = uses the summary to make predictions at new points

Different NP variants differ in how they encode, aggregate, and decode:

- CNP: Simple mean aggregation, point predictions
- NP: Adds a latent variable for uncertainty
- ANP: Adds attention for better predictions

## How It Works

Neural Processes (NPs) are a family of models that define a distribution over functions.
They encode a context set (observed points) into a representation and use it to make
predictions at target points. Unlike gradient-based meta-learners (MAML), NPs perform
adaptation through a single forward pass via amortized inference.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NeuralProcessBase(IFullModel<,,>,ILossFunction<>,IMetaLearnerOptions<>,IEpisodicDataLoader<,,>,IGradientBasedOptimizer<,,>,IGradientBasedOptimizer<,,>,Int32)` | Initializes shared NP infrastructure. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregateRepresentations(List<Vector<>>)` | Aggregates multiple context representations into a single representation via mean pooling. |
| `BuildContextRepresentations(Vector<>,Vector<>)` | Builds context representations from support features and labels by encoding each example as a context pair via `Vector{`. |
| `ComputeModScale(Vector<>)` | Computes a modulation scale from a representation vector using a shifted sigmoid that maps representation norm to [ModScaleBase, ModScaleBase + ModScaleRange]. |
| `DecodeTarget(Vector<>,Vector<>)` | Decodes a representation + target features into a prediction. |
| `EncodeContextPair(Vector<>,Vector<>)` | Encodes a single context pair (x, y) into a representation vector. |
| `InitializeParams(Int32)` | Initializes parameter vector with small random values. |
| `KLDivergenceGaussian(Vector<>,Vector<>,Vector<>,Vector<>)` | Computes the KL divergence between two Gaussian distributions parameterized by (mean1, logvar1) and (mean2, logvar2). |
| `ModulateParameters(Vector<>,Double)` | Modulates backbone parameters by a scale derived from the context representation, then sets them on the model. |
| `ReparameterizeSample(Vector<>,Vector<>)` | Samples from a Gaussian distribution using the reparameterization trick: z = mean + exp(0.5 * logvar) * epsilon, where epsilon ~ N(0,1). |
| `StandardNPAdapt(IMetaLearningTask<,,>)` | Standard NP adaptation: build context reps, aggregate, modulate, return adapted model. |
| `StandardNPMetaTrain(TaskBatch<,,>,Double)` | Standard NP meta-training loop shared by simple NP variants (EquivCNP, SwinTNP, TNP, etc.). |

## Fields

| Field | Summary |
|:-----|:--------|
| `DecoderParams` | Learned decoder parameters for mapping representations to predictions. |
| `EncoderParams` | Learned encoder parameters for mapping context pairs to representations. |
| `ModScaleBase` | Base value for the modulation scale sigmoid. |
| `ModScaleRange` | Range of the modulation scale sigmoid output. |
| `ModScaleSigmoidCenter` | Center of the sigmoid for modulation scale computation. |
| `RepresentationDim` | Dimensionality of the representation space. |

