# CodeXGLUE Benchmark Harness

AiDotNet includes a **CodeXGLUE harness** for evaluating code understanding / code generation models against a caller-provided dataset file (AiDotNet does not ship CodeXGLUE datasets).

## Dataset Format (JSONL)

Provide a JSONL file (one JSON object per line) with at least these fields:

- `source`: the prompt / input text (or source code, depending on the task)
- `target`: the expected output (or reference answer)

Optional fields:

- `id`: record identifier
- `category`: category label for per-category accuracy breakdown

Example line:

```json
{"id":"0","category":"code-to-text","source":"def add(a,b): return a+b","target":"Adds two numbers and returns the sum."}
```

## Metrics

The harness computes:

- Exact Match (after normalization)
- Token-level F1 (bag-of-tokens overlap)
- BLEU-4 (n-gram overlap with brevity penalty)
- ROUGE-L (longest common subsequence F1)
- Identifier-F1 (identifier token overlap)
- CodeBLEU-lite (heuristic aggregate of BLEU/ROUGE/TokenF1/IdentifierF1)

More task-specific metrics (BLEU/ROUGE/CodeBLEU) can be layered on in future improvements.

## Usage (Library)

Use `AiDotNet.Reasoning.Benchmarks.CodeXGlueBenchmark<T>` with `AiDotNet.Reasoning.Benchmarks.CodeXGlueBenchmarkOptions` and provide a model invocation function.
