# Program Synthesis Benchmarks (HumanEval + CodeXGLUE)

This document describes how to run program synthesis/code generation benchmarks using the façade APIs:
- Build/configure via `AiModelBuilder`
- Evaluate via `AiModelResult`
- Prefer `AiDotNet.Serving` for untrusted execution

## HumanEval (execution-based)

HumanEval evaluation is implemented as **execution-based** correctness:

- Preferred: execute via `AiDotNet.Serving` sandbox (`ExecuteProgramAsync`)
- Optional (dev only): local execution behind `AIDOTNET_HUMANEVAL_EXECUTION=1`

Recommended pattern:

1. Configure a model result with program-synthesis serving:
   - `AiModelBuilder.ConfigureProgramSynthesisServing(...)`
2. Use:
   - `AiModelResult.EvaluateHumanEvalAsync(...)` (pass@1 style)
   - `AiModelResult.EvaluateHumanEvalPassAtKAsync(k, ...)` (pass@k)

## CodeXGLUE

CodeXGLUE evaluation is dataset-driven and does not ship datasets in this repository.

1. Prepare a JSONL file (see `docs/reasoning/CodeXGLUE.md`).
2. Use `AiDotNet.Reasoning.Benchmarks.CodeXGlueBenchmark<T>` with `CodeXGlueBenchmarkOptions` and provide an `evaluateFunction`.

