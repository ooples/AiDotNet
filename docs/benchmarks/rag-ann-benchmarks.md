# AiDotNet RAG / ANN Vector-Search Benchmarks

A deterministic, seeded, CPU-only comparative benchmark of AiDotNet's in-repo
vector-search indexes and end-to-end retrieval path. The harness is a plain
`System.Diagnostics.Stopwatch`-based console app (no BenchmarkDotNet) so it stays
fast and runnable in CI or on a laptop.

- Harness: `benchmarks/AiDotNet.Rag.Benchmarks/` (net10.0 console, ProjectReferences `src/AiDotNet.csproj`)
- Run: `dotnet run --project benchmarks/AiDotNet.Rag.Benchmarks/AiDotNet.Rag.Benchmarks.csproj -c Release`
- Numbers below were produced on 2026-07-20 by a single run with `AIDOTNET_DISABLE_GPU=1`.

## Scope note (what is actually measured)

Two index families are measured on the same data:

- The managed in-memory indexes under `src/RetrievalAugmentedGeneration/VectorSearch/Indexes/`:
  `FlatIndex<T>` (exact — the recall ground truth), `IVFIndex<T>`, `HNSWIndex<T>`, `LSHIndex<T>`.
- The **native `AnnVectorIndex<T>`** (Flat / IVF / PQ / IVFPQ) built on the dependency-free
  `AiDotNet.Tensors` fused ANN kernels — the FaissNet replacement. **PQ / IVFPQ are only
  available here** (the managed set has none).

The end-to-end retrieval suite runs through `InMemoryDocumentStore<T>` (HNSW-backed).

> **Headline:** `NativeAnn IVF` reaches **recall@10 = 1.000** at **~54 µs p50 / 16.5k q/s**
> (1k) — beating managed IVF (0.925) on both recall and latency, and `NativeAnn Flat` is
> exact and faster than managed Flat. **However, `NativeAnn PQ`/`IVFPQ` recall is currently
> poor (0.04–0.45)** — a real defect under the cosine (inner-product) metric, tracked for a
> fix (see Interpretation). So the native stack already *exceeds* the managed indexes on the
> Flat/IVF path, but the compressed PQ path is **not yet** competitive with FAISS PQ.

## Methodology

**ANN suite (recall / latency / build).** For each corpus size N ∈ {1 000, 10 000}:

1. Generate `N` clustered embeddings, `dim = 128`, from `sqrt(N)` Gaussian cluster
   centers (per-component σ = 0.35), plus 200 disjoint query vectors sampled from the
   same clusters. All sampling is seeded (Box–Muller over a seeded `Random`), so the
   dataset is bit-identical across runs and machines.
2. Build an exact `FlatIndex` and take its top-10 per query as **ground truth**.
3. For each index type, measure:
   - **Recall@10** = mean over queries of `|approx_top10 ∩ groundtruth_top10| / 10`.
   - **Build+train (ms)** = bulk `AddBatch` + one warmup query. The warmup forces any
     lazy structure to materialize and be timed — notably `IVFIndex`'s k-means
     clustering, which is built lazily on first search.
   - **Latency p50 / p95 (µs)** and **throughput (q/s)** = per-query `Search(q, 10)`
     wall time over 3 repeated passes of the 200-query set (600 samples), nearest-rank
     percentiles.

**End-to-end retrieval suite.** A small fixed, hand-labeled corpus (20 documents
across 5 topics: Python, cooking, astronomy, personal finance, gardening) with a known
query→relevant-doc mapping (5 queries, 4 relevant docs each). Documents and queries are
embedded with a deterministic dependency-free bag-of-words embedder (FNV-1a token
hashing into a 128-d vector, L2-normalized), then indexed and queried through
`InMemoryDocumentStore<double>`. We report **Recall@5, Recall@10, MRR, nDCG@10**
(binary relevance) and retrieval-call latency. Real **RAGAS** metrics
(faithfulness / answer-relevancy) are intentionally **skipped**: they require a live
LLM judge, which is out of scope for a deterministic offline harness.

**Determinism.** All randomness is seeded (`SyntheticData` master seed `20240719`;
`IVFIndex`/`HNSWIndex`/`LSHIndex` use their built-in seed 42). Recall / IR-quality
numbers are exactly reproducible run to run (verified across 3 runs). Build and latency
figures are single-run wall-clock and carry normal machine noise (see caveats).

## Hardware / runtime

| | |
|---|---|
| OS | Microsoft Windows 11 (10.0.22631), X64 |
| CPU | 16 logical cores, x64 |
| Runtime | .NET 10.0.10 |
| GC | Server GC enabled |
| Config | dim=128, k=10, 200 queries, 3 latency passes |

**CPU-only, no GPU, no external vector DB.** The vector indexes are pure managed C#
(dictionary storage + cosine similarity over `Vector<double>`); no GPU kernels and no
external vector database (FAISS/Pinecone/Qdrant/etc.) are exercised. Note that the
`AiDotNet.Tensors` package auto-initializes an OpenCL context at load if a device is
present; this run set `AIDOTNET_DISABLE_GPU=1` to guarantee a clean CPU-only path (the
dim-128 dot products run on the managed CPU engine regardless).

## Results — ANN indexes

### N = 1,000 vectors, dim = 128

| Index | Recall@10 | Build+train (ms) | p50 (µs) | p95 (µs) | Throughput (q/s) |
|-------|-----------|------------------|----------|----------|------------------|
| Flat (exact) | 1.000 | 1.2 | 670.7 | 1749.5 | 1,169 |
| IVF (nlist=31, nprobe=3) | 0.925 | 82.0 | 156.7 | 263.8 | 2,670 |
| HNSW (M=16, efC=200, efS=50) | 0.980 | 2037.6 | 137.8 | 341.6 | 6,098 |
| LSH (tables=10, bits=12) | 0.802 | 24.0 | 32.8 | 44.9 | 25,587 |
| **NativeAnn Flat (exact)** | **1.000** | 22.9 | 399.4 | 592.9 | 947 |
| **NativeAnn IVF (nlist=31, nprobe=3)** | **1.000** | 53.2 | **54.5** | 83.0 | **16,568** |
| NativeAnn PQ (m=8, ksub=250) | 0.454 | 291.7 | 119.1 | 132.8 | 8,354 |
| NativeAnn IVFPQ (nlist=31, nprobe=3, m=8, ksub=250) | 0.085 | 400.3 | 136.6 | 189.9 | 7,010 |

### N = 10,000 vectors, dim = 128

| Index | Recall@10 | Build+train (ms) | p50 (µs) | p95 (µs) | Throughput (q/s) |
|-------|-----------|------------------|----------|----------|------------------|
| Flat (exact) | 1.000 | 13.7 | 9901.8 | 15353.1 | 98 |
| IVF (nlist=100, nprobe=12) | 0.990 | 856.1 | 1298.5 | 2899.6 | 670 |
| HNSW (M=16, efC=200, efS=50) | 0.700 | 17176.7 | 254.1 | 552.7 | 3,294 |
| LSH (tables=10, bits=12) | 0.817 | 90.4 | 155.0 | 269.1 | 5,795 |
| **NativeAnn Flat (exact)** | **1.000** | 42.4 | 6137.4 | 11406.7 | 145 |
| **NativeAnn IVF (nlist=100, nprobe=12)** | **1.000** | 1546.2 | **621.8** | 1110.3 | **1,217** |
| NativeAnn PQ (m=8, ksub=256) | 0.143 | 3162.5 | 1007.5 | 1126.7 | 979 |
| NativeAnn IVFPQ (nlist=100, nprobe=12, m=8, ksub=256) | 0.041 | 4684.6 | 504.1 | 579.8 | 1,968 |

Ground truth is `FlatIndex` (recall 1.000 by definition). Recall figures are exactly
reproducible; build/latency are single-run and noisy (see caveats).

## Results — end-to-end retrieval

`InMemoryDocumentStore<double>` (HNSW-backed), deterministic stub embedder, 20 labeled
docs, 5 labeled queries.

| Recall@5 | Recall@10 | MRR | nDCG@10 | p50 (µs) | p95 (µs) |
|----------|-----------|-----|---------|----------|----------|
| 0.600 | 0.750 | 1.000 | 0.742 | 19.6 | 42.2 |

MRR = 1.000 means the single most-relevant document for every query ranked #1.
Recall@10 = 0.750 reflects that with only 20 docs and topK=10, some of a topic's 4
relevant docs share fewer distinctive keywords and fall outside the top-10 under the
bag-of-words embedder — this is a property of the deliberately simple stub embedder,
not an index defect.

## Interpretation

- **The native `AnnVectorIndex` Flat/IVF path leads the field measured here.**
  `NativeAnn Flat` is exact (recall 1.000) and lower-latency than managed Flat
  (399 vs 671 µs p50 @1k; 6.1 vs 9.9 ms @10k). `NativeAnn IVF` is the standout:
  **recall 1.000 at 54 µs p50 / 16.5k q/s (1k)** and **1.000 at 622 µs / 1.2k q/s (10k)** —
  higher recall *and* lower latency than the managed `IVFIndex` (0.925 / 0.990). This is the
  dependency-free FaissNet replacement beating the prior in-house indexes.
- **`NativeAnn PQ` / `IVFPQ` recall is currently a defect (0.04–0.45), not just lossy
  compression.** Under the cosine (inner-product) metric the PQ codebooks are trained with
  Lloyd's k-means whose assignment uses the metric but whose centroid update is the
  Euclidean mean — consistent for L2, **inconsistent for inner product** — so on
  L2-normalized vectors the sub-quantizers partition poorly and ADC ranking collapses
  (IVFPQ at 10k ≈ 0.04, near-random). FAISS PQ reaches ~0.7–0.95 on comparable data, so
  **this path does not yet meet — let alone exceed — the standard.** Fix direction: train PQ
  codebooks under L2 even when the user selects cosine (for unit vectors L2-nearest ≡
  cosine-nearest, making Lloyd's consistent), or use spherical k-means for the IP metric.
  Tracked as a follow-up in `AiDotNet.Tensors` (`AnnIndex.TrainPq`).
- **Recall / speed trade-off holds as expected.** Flat is exact but its per-query cost
  grows linearly. LSH is the fastest per query and cheapest to build, at the lowest recall
  (~0.80). Managed IVF gives a strong recall/latency balance (0.925–0.990) with moderate
  build cost.
- **HNSW recall regresses at 10k with default parameters** (0.980 at 1k → **0.700** at
  10k). With `efSearch=50` (the `InMemoryDocumentStore` default) the graph beam search
  does not recover enough of the true top-10 as the graph grows, and HNSW build is by
  far the most expensive step (~1.4 s at 1k, ~8–31 s at 10k across runs). This is a
  genuine, reproducible finding: for higher recall at scale this HNSW implementation
  needs a larger `efSearch`/`efConstruction`, at additional build/query cost.
- **IVF is the sweet spot at 10k here** — 0.990 recall at ~630 q/s, ~8× the Flat
  throughput, with a ~1.5 s build.

## How these compare to industry (FAISS / hnswlib / LangChain)

This is an **honest, indirect** framing — the harness does **not** run FAISS, hnswlib,
or LangChain, so nothing below is a head-to-head measurement.

- **Absolute latency.** Mature native libraries (FAISS, hnswlib) are C/C++ with SIMD
  and cache-tuned layouts; their per-query latencies for these sizes are typically in
  the low-single-digit to tens of microseconds. AiDotNet's managed indexes here are in
  the tens-to-hundreds of microseconds (ANN) — same order of magnitude for LSH/HNSW,
  but generally slower, as expected for a pure-managed implementation without a packed
  contiguous vector buffer or SIMD distance kernels.
- **Recall.** The recall/latency *shape* matches the literature: exact ≥ HNSW ≈ IVF ≥
  LSH at fixed effort. hnswlib at `efSearch` in the hundreds routinely hits >0.95
  recall@10 at 10k–1M; AiDotNet's HNSW would need larger `ef` to match, and its recall
  drop at 10k with default `ef` is a real gap versus hnswlib's tuned defaults.
- **Build time.** FAISS/hnswlib build 10k×128 in well under a second; AiDotNet's HNSW
  build (seconds) is materially slower — the biggest relative gap in this benchmark.
- **LangChain.** LangChain is an orchestration layer, not an index; it delegates to
  FAISS/Chroma/etc. The relevant comparison is the *end-to-end retrieval quality*
  (recall/MRR/nDCG), which is embedder-dominated. Our stub embedder is intentionally
  trivial, so the IR numbers here characterize the harness, not a production embedder.
- **Not measured:** PQ/IVFPQ memory compression (no PQ index exists in-repo), SIMD/GPU
  distance kernels, disk-backed or sharded indexes, and concurrent multi-client
  throughput.

**Takeaway:** the AiDotNet managed indexes reproduce the correct algorithmic
trade-offs and are fast enough for small/medium in-process RAG, but the pure-managed
implementations trail specialized native ANN libraries on absolute latency and
(especially) build time, and the HNSW defaults under-recall at 10k.

## Caveats

- Single-machine, single-run build/latency numbers with normal OS scheduling noise.
  Across three runs, **recall / MRR / nDCG were bit-identical**, while HNSW 10k build
  time varied from ~6 s to ~31 s and per-query latencies varied ~1.5–3× — treat the
  timing columns as indicative orders of magnitude, not precise constants.
- Only two corpus sizes (1k, 10k) and one dimension (128). No million-scale test.
- Cosine similarity only; other metrics (Euclidean/dot/Manhattan/Jaccard) exist in the
  repo but were not swept.
- RAGAS faithfulness/answer-relevancy skipped (requires a live LLM judge).

## Reproducing

```bash
# From repo root:
AIDOTNET_DISABLE_GPU=1 dotnet run \
  --project benchmarks/AiDotNet.Rag.Benchmarks/AiDotNet.Rag.Benchmarks.csproj -c Release
```

The program prints the environment block and all tables above as GitHub-flavored
markdown to stdout; progress messages go to stderr.
