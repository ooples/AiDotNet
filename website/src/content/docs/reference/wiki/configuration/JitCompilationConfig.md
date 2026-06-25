---
title: "JitCompilationConfig"
description: "Configuration for JIT (Just-In-Time) compilation of model forward/backward passes."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Configuration`

Configuration for JIT (Just-In-Time) compilation of model forward/backward passes.

## For Beginners

JIT compilation is like building a shortcut for your
model. The first time you call `Predict` the library watches which math
operations happen in what order, compiles that pattern into a flat fast
pipeline, and from then on just replays the pipeline. Your model doesn't
change — only the plumbing around it gets faster.

## How It Works

JIT compilation traces the model's computation graph on the first call and
replays the compiled plan on subsequent calls, eliminating virtual dispatch,
per-op allocation, and bounds-checking overhead. Typical speedup is 1.5-3x on
CPU and up to 10x on GPU for small batches where dispatch dominates.

Under the hood this binds to the `TensorCodecOptions` in the Tensors
package. The builder writes the options into the thread-local codec config
before the built model sees any work; `Tensor{`
and the tape-based training path then route through their compiled variants
via `CompiledModelCache`.

When to enable:

- Production inference where Predict is called many times at the same shape.
- Training loops where the forward+backward is invoked each iteration.
- Diffusion/autoregressive generation where a sub-network runs tens of times per call.

When to disable / when to keep `ThrowOnFailure` off:

- Debugging a model whose forward path has non-Engine tensor accesses

(direct span writes, scalar control flow) — those bake at trace time and will
replay stale data. The default fallback to eager execution hides this.

- Tiny models where the compile cost isn't amortized (rare — overhead is

one traced forward).

Example YAML (binds through `YamlConfigApplier`):

Example code:

## Properties

| Property | Summary |
|:-----|:--------|
| `Aggressive` | Aggressive config — all fusion and constant-folding passes on. |
| `DataflowFusionMaxHidden` | Maximum hidden dimension for dataflow fusion L1 residency. |
| `Default` | Default config — compilation enabled, failures fall back silently to eager. |
| `Disabled` | Explicitly disabled — compiled paths short-circuit to eager. |
| `EnableAlgebraicBackward` | Phase C: Symbolically simplify the backward graph at compile time (CSE, double-transpose elimination, associative regrouping). |
| `EnableAttentionFusion` | Phase 4.2: Fuse attention Q@K^T → Softmax → V patterns. |
| `EnableBlasBatch` | Phase 7.1: Group independent MatMuls into batched calls. |
| `EnableConstantFolding` | Phase 4.5: Precompute static subgraphs at compile time. |
| `EnableConvBnFusion` | Phase 4.1: Fold BatchNorm into Conv2D weights at compile time. |
| `EnableDataflowFusion` | Phase B: Fuse consecutive linear layers into a single multi-layer kernel that keeps data in registers / L1 across layer boundaries. |
| `EnableForwardCSE` | Phase 6.2: Deduplicate identical computations across layers. |
| `EnableMixedPrecisionCompilation` | Phase 7.3: Mixed precision (fp16 forward, fp32 backward). |
| `EnablePointwiseFusion` | Phase 4.3: Merge consecutive pointwise ops into fewer dispatch steps. |
| `EnableSpectralDecomposition` | Phase A: SVD-factorize frozen weight matrices for faster inference. |
| `Enabled` | Master switch. |
| `SpectralErrorTolerance` | Maximum approximation error per element for spectral decomposition (used as `energyThreshold = 1.0 - tolerance` for SVD rank selection). |
| `ThrowOnFailure` | When `true`, a compilation failure in the builder's `JitCompiledFunction` wrapper (the one populated by `AiModelBuilder.BuildCompiledPredictFunction`) propagates as an exception instead of silently falling back to the eager Predict path. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyToTensorCodec` | Projects this config onto the Tensors-package `TensorCodecOptions` and installs it as the current thread-local config. |
| `Validate` | Validates the config and throws if settings are inconsistent. |

