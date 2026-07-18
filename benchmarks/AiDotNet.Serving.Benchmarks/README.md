# AiDotNet Serving Benchmark (`adnbench`)

A backend-agnostic load generator for LLM serving endpoints. It measures the metrics the
industry competes on — **throughput, TTFT, inter-token latency, end-to-end latency, and
goodput** — under an identical, reproducible workload, so AiDotNet can be compared *apples-to-apples*
against vLLM, TGI, and TensorRT-LLM.

> **Why this exists first.** You cannot claim to "exceed vLLM / TensorRT / TGI" without a
> reproducible number that says so. This harness turns that goal from a slogan into a metric
> tracked per change.

## Design

- **`--backend openai`** — drives `/v1/chat/completions` or `/v1/completions` with `stream=true`
  and parses the SSE token stream. Works against **vLLM**, **TGI** (OpenAI route), and **AiDotNet**,
  which now exposes the OpenAI API (`/v1/chat/completions`, `/v1/completions`) via this serving layer.
  This is the common denominator that makes comparisons fair.
- **`--backend aidotnet-native`** — drives AiDotNet's `api/inference/generate/{model}`
  endpoint (token-ID native, **non-streaming**). Use it to measure the engine's raw throughput and
  end-to-end latency. TTFT / ITL / TPOT are unavailable on this path because the endpoint returns the
  whole completion at once — prefer the streaming `--backend openai` route for those latency metrics.

The load driver honors a **Poisson arrival schedule** (`--request-rate`) bounded by a
**concurrency cap** (`--concurrency`), the same model vLLM's `benchmark_serving.py` uses.

## Build & run

```bash
# from repo root, using the pinned SDK
dotnet run -c Release --project benchmarks/AiDotNet.Serving.Benchmarks -- --help
```

### Measure AiDotNet's engine today (native endpoint)

```bash
dotnet run -c Release --project benchmarks/AiDotNet.Serving.Benchmarks -- \
  --backend aidotnet-native --base-url http://localhost:5000 --model my-llm \
  --num-prompts 500 --concurrency 32 --input-tokens 256 --output-tokens 128
```

### Compare against vLLM

```bash
dotnet run -c Release --project benchmarks/AiDotNet.Serving.Benchmarks -- \
  --base-url http://localhost:8000 --model meta-llama/Llama-3.1-8B-Instruct \
  --num-prompts 500 --request-rate 10 --output-json vllm.json
```

### Compare against TGI (OpenAI route)

```bash
dotnet run -c Release --project benchmarks/AiDotNet.Serving.Benchmarks -- \
  --base-url http://localhost:8080 --model tgi --mode chat \
  --num-prompts 500 --request-rate 10 --output-json tgi.json
```

Run the same command against each server (same `--seed`, `--num-prompts`, `--request-rate`,
`--input-tokens`, `--output-tokens`) and diff the JSON reports.

## Metrics

| Metric | Meaning |
|---|---|
| **Request throughput** | Completed requests / sec |
| **Output throughput** | Generated tokens / sec (the headline serving number) |
| **Total token throughput** | (prompt + output) tokens / sec |
| **TTFT** | Time to first token (streaming only) — prefill responsiveness |
| **TPOT** | Mean time per output token per request (decode speed) |
| **ITL** | Inter-token latency across all tokens (decode smoothness / p99 stalls) |
| **E2E** | Full request latency |
| **Goodput** | Requests/sec meeting both the TTFT and TPOT SLAs |

## Notes & honesty caveats

- Synthetic prompts approximate **1 word ≈ 1 token**; for real length distributions pass
  `--dataset prompts.txt` (one prompt per line). Token *counts* in the report come from the server's
  `usage` when it provides it (vLLM does via `stream_options.include_usage`), otherwise from counting
  streamed content chunks (an approximation).
- The native backend synthesizes random token IDs in `[1, --vocab)`; it exercises the engine, not a
  real tokenizer/prompt distribution.
- Warmup requests (`--warmup`) are excluded from the reported metrics.
- This tool measures a running server over HTTP; it does not stand one up. Start the server first.
