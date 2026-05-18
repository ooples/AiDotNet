# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2025-12-18

### Breaking changes (PR #1368)

- **`AiModelBuilder<T,TInput,TOutput>`'s test-only `Configured*` accessors
  moved behind an explicit-interface implementation.** PR #1368 extracted
  the 8 `internal Configured*` properties on `AiModelBuilder` into an
  explicitly-implemented `internal interface IConfiguredView<T,TInput,TOutput>`
  to keep them off the regular type surface. **Migration for other
  `InternalsVisibleTo` consumers:** cast the builder to
  `AiDotNet.Configuration.IConfiguredView<T,TInput,TOutput>` before
  accessing `ConfiguredOptimizer`, `ConfiguredCaching`,
  `ConfiguredInferenceOptimizations`, `ConfiguredJitCompilation`,
  `ConfiguredInterpretability`, `ConfiguredMemoryManagement`,
  `ConfiguredLicenseKey`, or `ConfiguredAgentAssistance`. Source:
  `src/Configuration/IConfiguredView.cs`, `src/AiModelBuilder.cs` (review
  #1368 C8eN_).

- **`AiModelResult.Predict` now throws `InvalidOperationException` with a
  clear diagnostic when `Model` is null.** Previously a null Model would
  surface as a `NullReferenceException` deep inside the dispatch path.
  The new path throws "AiModelResult.Model is null; cannot dispatch
  Predict." at the call site (review #1368 C8eRj). **Migration:** if any
  caller was catching `NullReferenceException` to detect "no model
  configured", switch to catching `InvalidOperationException` or check
  `result.Model is not null` before calling Predict.

- **`ConfigureRegularization` now throws when paired with a non-gradient optimizer.**
  Previously, calling `ConfigureRegularization(...)` while the active optimizer
  was `NormalOptimizer` / an evolutionary search / any custom `IOptimizer`
  outside the `GradientBasedOptimizerBase` family silently dropped the
  configured regularization (the gradient-application loop where the slot is
  consumed never runs on non-gradient optimizers, so the value was stored on
  the builder and forgotten). PR #1368 surfaces this as a `Build`-time
  `InvalidOperationException` so the misconfiguration is caught before training
  starts. **Migration:** either remove the `ConfigureRegularization(...)` call
  on builds that use a non-gradient optimizer, or switch the optimizer to a
  `GradientBasedOptimizerBase` subclass (`AdamOptimizer`, `SGDOptimizer`,
  `AdamWOptimizer`, etc.). Source: `src/AiModelBuilder.cs:2699-2719`.

- **`LoRAAdapterBase.CreateLoRALayer` now throws when neither weight-matrix
  probing nor the shape API can resolve both `inputSize` and `outputSize`.**
  Previously the adapter silently produced a wrongly-sized LoRA layer using
  `outSize * 2` as a heuristic fallback, which later crashed at forward time
  with an opaque shape mismatch. Now the throw surfaces the unresolvable layer
  at the LoRA wrap step where the upstream cause is obvious. **Migration:**
  ensure LoRA-wrapped layers have their shape resolved (run one warmup forward
  through the model before `ConfigureLoRA(...)` fires, or supply a layer that
  exposes authoritative `GetInputShape()` values) before the wrap loop fires.
  Source: `src/LoRA/Adapters/LoRAAdapterBase.cs`.

- **`AiModelResult` constructor throws on unfitted `PostprocessingPipeline`.**
  Previously the unfitted pipeline rode through to `AiModelResult.Predict` and
  threw on the first inference call — surfacing the misconfiguration at the
  wrong layer of the stack. The constructor now fail-fasts at Build time.
  **Migration:** either fit `PostprocessingPipeline.Fit(...)` before passing
  the options to the ctor, or set
  `AiModelResultOptions.PostprocessingFitSample` to a representative
  training-target sample so the ctor can lazy-fit (review #1368 C6WMS).
  `AiModelBuilder` build paths already do this — the breaking change only
  affects callers that construct `AiModelResultOptions` manually.

- **Inference fast paths (`InferenceOptimizationConfig`, `JitCompiledFunction`)
  now flow through postprocessing and the safety filter.** Previously these
  branches returned the raw model output early, bypassing the
  `denormalize → PostprocessingPipeline.Transform → SafetyFilter` tail. Now
  `DispatchModelInference` returns into a single shared tail that all three
  paths (optimized, JIT, standard) traverse. **Migration:** if downstream
  consumers relied on the fast paths skipping postprocessing for performance,
  add `.ConfigurePostprocessing(null)` / leave the pipeline unconfigured —
  the shared tail short-circuits when no postprocessing is wired. Source:
  `src/Models/Results/AiModelResult.cs:DispatchModelInference` (review #1368
  C7HAa).

- **`ConfigureKnowledgeDistillation` on the regular-training (non-LoRA, non-
  direct-training) NN path now throws `NotSupportedException` at Build time.**
  Previously the second throw site was on this path; the option was stored but
  never reached the tape-based training flow. The throw is preserved (not
  downgraded to a Trace-warning) so users discover the missing KD-tape
  integration at Build time rather than getting standard supervised training
  silently substituted for the requested distillation. **Migration:** either
  configure LoRA / a direct-training path (where the options ARE attached to
  the result without going through the regular tape loop), or wait for the
  upstream KD-tape integration to land. Source: `src/AiModelBuilder.cs:3636`.

### Added
- **iMAML (implicit Model-Agnostic Meta-Learning) Algorithm**
  - True implicit differentiation implementation with constant memory complexity
  - Configurable Hessian-vector product computation (finite differences, automatic differentiation)
  - Preconditioned Conjugate Gradient solver with multiple strategies
  - Adam-style adaptive learning rates for inner loop optimization
  - Optional line search for optimal step size finding
  - Comprehensive configuration options for production use

- **SEAL Algorithm Enhancements**
  - Production-ready adaptive learning rate feature with multiple strategies (Adam, RMSProp, Adagrad, GradNorm)
  - Second-order approximation with full backpropagation through adaptation steps
  - Per-parameter adaptive learning rate tracking with warmup and clamping

- **MAML Algorithm Performance Optimization**
  - Fixed redundant adaptation performance issue
  - Eliminated double computation per task
  - Added adaptation history tracking to avoid repeated gradients

### Documentation
- Added comprehensive production-ready PR process guide
- Created iMAML usage guide with examples and best practices
- Updated PR checklist template for future development

### Testing
- Added unit tests for iMAML algorithm with >90% coverage
- Created performance benchmarks comparing iMAML configurations
- Added memory usage benchmarks validating O(N) space complexity

### Security
- No hardcoded secrets or credentials
- Proper input validation throughout implementations
- Secure random number generation using project's RandomHelper

---

## Previous Versions

[Previous changelog entries would appear here]

## [0.0.1] - 2023-XX-XX

### Added
- Initial project setup
- Basic neural network implementation
- Core algorithms and data structures