# Program Synthesis Scripts

This folder contains helper notes for running program synthesis benchmarks locally.

## Prerequisites

- `.NET SDK` installed
- Optional (recommended): run `AiDotNet.Serving` locally for sandboxed execution

## Serving (recommended)

Run Serving from `src/AiDotNet.Serving` and configure:
- `ServingSandbox` tier limits (execution safety)
- `ServingProgramSynthesis` tier limits (code tasks safety)

## HumanEval

Use `AiModelResult.EvaluateHumanEvalAsync(...)` or `AiModelResult.EvaluateHumanEvalPassAtKAsync(k, ...)`.

Local execution is intentionally opt-in only:
- Set `AIDOTNET_HUMANEVAL_EXECUTION=1` for dev/test environments.

To evaluate against a local HumanEval JSONL dataset (without shipping it in-repo):
- Set `AIDOTNET_HUMANEVAL_DATASET=<path-to-HumanEval.jsonl>`

## Tooling (recommended)

For reproducible training/evaluation runs without adding new public library APIs, use:

- `dotnet run --project tools/AiDotNet.ProgramSynthesis.Tooling -- train --train <train.jsonl> --output <model.model>`
- `dotnet run --project tools/AiDotNet.ProgramSynthesis.Tooling -- evaluate --model <model.model> --codeXGlue <dataset.jsonl> --report <report.json>`
