Project Coding Rules (authoritative summary)

Architecture
- Integrate with `PredictionModelBuilder<T,TInput,TOutput>` and existing interfaces where applicable.
- Use project numeric abstractions: `INumericOperations<T>` via `MathHelper.GetNumericOperations<T>()`.
- Use project collections: `Vector<T>`, `Matrix<T>`, `Tensor<T>` — avoid arrays in public APIs.

Generics & Types
- All algorithms generic on `T` (numeric type). No hardcoded `double` in algorithms; convert constants via `INumericOperations<T>.FromDouble` as needed.
- Maintain .NET Framework 4.6.2 compatibility.

Code Organization
- One type per file (classes, interfaces, enums).
- Namespaces follow existing structure (e.g., `AiDotNet.Diffusion.Schedulers`).
- SOLID and DRY: prefer small, composable units; avoid duplication; inject dependencies where relevant.

Testing & Coverage
- xUnit tests for new code, target `net462` and `net8.0`.
- Minimum 90% coverage for new/changed code; use `coverlet.collector`.

PR Process
- Conventional Commits in messages.
- Link issues with `Fixes #<num>`.
- Ensure builds/tests pass locally for both target frameworks.
