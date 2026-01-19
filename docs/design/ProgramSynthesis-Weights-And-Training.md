# Program Synthesis: Weights, Training, and Deployment

This document describes a production-friendly way to handle **weights** for code models and program synthesis without shipping proprietary artifacts in the repository.

## Weights Strategy (Production)

AiDotNet code models (e.g., CodeBERT / GraphCodeBERT / CodeT5) and synthesizers are `IFullModel` implementations and support serialization through the existing model serialization APIs.

Recommended approach:

1. Train or fine-tune the model in a controlled environment.
2. Export the trained model as a serialized model file.
3. Deploy the serialized file to your Serving environment via secure artifact distribution (CI/CD, storage with access controls).
4. Configure `AiDotNet.Serving` to load the model at startup (or provision it via your model repository implementation).

This approach:
- Avoids embedding weights in source control.
- Keeps the public API surface minimal (users interact via `AiModelBuilder` + `AiModelResult`; Serving executes untrusted code).
- Supports monetization/IP constraints by keeping proprietary weights private.

## Model Serialization (Library)

`AiModelResult` supports saving and loading serialized models:

- Save: `AiModelResult.SaveToFile(...)`
- Load: `AiModelResult.LoadFromFile(...)`

These APIs store architecture + parameters via the model’s `Serialize()` implementation.

## Training / Fine-Tuning (Current State)

The library has the scaffolding to represent and run code tasks end-to-end, but **competitive pretrained weights** require:
- A dataset ingestion pipeline for code tasks
- Tokenization and batching consistent with the target model family
- A reproducible training loop and evaluation harness

For “100% confidence” on Issue #404 and “exceeds industry standards”, the repository should add:
- A reproducible training recipe (scripts + docs)
- Clear dataset format expectations and preprocessing steps
- A benchmark report for HumanEval and at least one CodeXGLUE task family

## Serving Deployment Notes

For untrusted execution and correctness evaluation:
- Prefer Serving-first execution via the sandbox endpoints (`/api/program-synthesis/program/*`).
- Keep local execution paths opt-in only (dev/test environments).

For reproducible model deployments:
- Use `ServingOptions.StartupModels[].Sha256` to enforce a SHA-256 integrity check of model files at startup.

For request safety:
- Enforce tier-scoped limits (size + concurrency + timeouts).
