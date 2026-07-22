#!/usr/bin/env bash
# ===========================================================================
# Serving head-to-head: AiDotNet vs vLLM / TGI / SGLang / TensorRT-LLM.
#
# All four competitors expose an OpenAI-compatible /v1 API, so the SAME
# benchmark driver (adnbench, this repo's benchmarks/AiDotNet.Serving.Benchmarks)
# measures every one of them identically: throughput (req/s, output tok/s,
# total tok/s), TTFT, TPOT/inter-token latency, end-to-end latency (mean/p50/
# p90/p99), and goodput vs an SLA. Each backend writes a JSON report; the final
# `adnbench compare` step renders a side-by-side table with ratios.
#
# REQUIREMENTS for a MEANINGFUL comparison (must be identical across backends):
#   * the SAME model (e.g. meta-llama/Llama-3.1-8B-Instruct),
#   * the SAME GPU/host,
#   * the SAME workload (--input-tokens/--output-tokens/--num-prompts/--concurrency).
# The competitors require an NVIDIA GPU + Linux (vLLM/TGI/SGLang/TensorRT-LLM are
# CUDA-only). AiDotNet must serve the SAME model — load it via the HF-safetensors
# import path (see the serving docs) rather than the synthetic DevHost model, or
# the comparison is apples-to-oranges (DevHost measures engine overhead only).
#
# Usage:
#   MODEL=meta-llama/Llama-3.1-8B-Instruct ./serving-head-to-head.sh
# Set BACKENDS to a subset, e.g. BACKENDS="aidotnet vllm" to skip some.
# ===========================================================================
set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
BACKENDS="${BACKENDS:-aidotnet vllm tgi sglang trtllm}"
NUM_PROMPTS="${NUM_PROMPTS:-256}"
CONCURRENCY="${CONCURRENCY:-32}"
INPUT_TOKENS="${INPUT_TOKENS:-512}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-128}"
WARMUP="${WARMUP:-16}"
OUT="${OUT:-./bench-results}"
ADNBENCH="dotnet run --project $(dirname "$0")/AiDotNet.Serving.Benchmarks/AiDotNet.Serving.Benchmarks.csproj -c Release --"

mkdir -p "$OUT"

# adnbench <label> <base-url> — run the standard workload against an OpenAI /v1 endpoint.
run_bench () {
  local label="$1" base="$2"
  echo "### benchmarking $label at $base"
  $ADNBENCH --backend openai --base-url "$base" --model "$MODEL" \
    --num-prompts "$NUM_PROMPTS" --concurrency "$CONCURRENCY" \
    --input-tokens "$INPUT_TOKENS" --output-tokens "$OUTPUT_TOKENS" --warmup "$WARMUP" \
    --output-json "$OUT/$label.json"
}

# ---- Reference commands to start each server (uncomment / adapt to your box) ----
# AiDotNet   : serve the imported HF model on :8001 (OpenAI route). See serving docs for the loader.
# vLLM       : vllm serve "$MODEL" --port 8002
# TGI        : docker run --gpus all -p 8003:80 ghcr.io/huggingface/text-generation-inference --model-id "$MODEL"
#              (TGI's OpenAI route is /v1 on that port)
# SGLang     : python -m sglang.launch_server --model-path "$MODEL" --port 8004
# TensorRT   : trtllm-serve "$MODEL" --port 8005   (or the Triton OpenAI frontend)

declare -A URL=(
  [aidotnet]="${AIDOTNET_URL:-http://localhost:8001}"
  [vllm]="${VLLM_URL:-http://localhost:8002}"
  [tgi]="${TGI_URL:-http://localhost:8003}"
  [sglang]="${SGLANG_URL:-http://localhost:8004}"
  [trtllm]="${TRTLLM_URL:-http://localhost:8005}"
)

labels=()
for b in $BACKENDS; do
  run_bench "$b" "${URL[$b]}"
  labels+=("$b=$OUT/$b.json")
done

echo
echo "### side-by-side comparison (baseline = first backend listed)"
$ADNBENCH compare "${labels[@]}"
