# AiDotNet RAG / ANN Vector-Search Benchmarks

A deterministic, seeded, CPU-only comparative benchmark of AiDotNet's in-repo
vector-search indexes and end-to-end retrieval path. The harness is a plain
`System.Diagnostics.Stopwatch`-based console app (no BenchmarkDotNet) so it stays
fast and runnable in CI or on a laptop.

- Harness: `benchmarks/AiDotNet.Rag.Benchmarks/` (net10.0 console, ProjectReferences `src/AiDotNet.csproj`)
- Run: `dotnet run --project benchmarks/AiDotNet.Rag.Benchmarks/AiDotNet.Rag.Benchmarks.csproj -c Release`
- Numbers below were produced on 2026-07-20 by a single run with `AIDOTNET_DISABLE_GPU=1`.

## Scope note (what is actually measured)

The original task referenced a native `AnnVectorIndex<T>` (Flat / IVF / PQ / IVFPQ)
and a `VectorIndexDocumentStore`. **Those types do not exist in this worktree** at the
benchmarked commit — the RAG vector-search surface present under
`src/RetrievalAugmentedGeneration/VectorSearch/Indexes/` is:

- `FlatIndex<T>` — exact brute-force search (used here as the recall ground truth)
- `IVFIndex<T>` — inverted-file / coarse-quantization ANN (random-seed centroids)
- `HNSWIndex<T>` — hierarchical navigable small-world graph ANN
- `LSHIndex<T>` — random-projection locality-sensitive hashing ANN

There is **no Product-Quantization (PQ) or IVFPQ** index in the codebase, so those
columns are omitted rather than fabricated. Per the task constraints, no `src/` code
was modified; the benchmark exercises the indexes exactly as shipped. The end-to-end
retrieval suite runs through `InMemoryDocumentStore<T>` (which is HNSW-backed
internally) since that is the concrete native/in-memory store available.

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
| Flat (exact) | 1.000 | 0.6 | 458.0 | 850.0 | 1,970 |
| IVF (nlist=31, nprobe=3) | 0.925 | 58.5 | 100.3 | 163.4 | 8,999 |
| HNSW (M=16, efC=200, efS=50) | 0.980 | 1434.1 | 120.7 | 206.7 | 7,739 |
| LSH (tables=10, bits=12) | 0.802 | 25.1 | 31.4 | 41.9 | 30,419 |

### N = 10,000 vectors, dim = 128

| Index | Recall@10 | Build+train (ms) | p50 (µs) | p95 (µs) | Throughput (q/s) |
|-------|-----------|------------------|----------|----------|------------------|
| Flat (exact) | 1.000 | 14.4 | 6055.7 | 45357.5 | 75 |
| IVF (nlist=100, nprobe=12) | 0.990 | 1464.7 | 1039.5 | 1801.5 | 627 |
| HNSW (M=16, efC=200, efS=50) | 0.700 | 27842.4 | 323.8 | 383.9 | 2,329 |
| LSH (tables=10, bits=12) | 0.817 | 263.9 | 171.6 | 374.4 | 2,013 |

Ground truth is `FlatIndex` (recall 1.000 by definition). Recall figures are exactly
reproducible; build/latency are single-run and noisy (see caveats).

## Results — end-to-end retrieval

`InMemoryDocumentStore<double>` (HNSW-backed), deterministic stub embedder, 20 labeled
docs, 5 labeled queries.

| Recall@5 | Recall@10 | MRR | nDCG@10 | p50 (µs) | p95 (µs) |
|----------|-----------|-----|---------|----------|----------|
| 0.600 | 0.750 | 1.000 | 0.742 | 46.5 | 112.1 |

MRR = 1.000 means the single most-relevant document for every query ranked #1.
Recall@10 = 0.750 reflects that with only 20 docs and topK=10, some of a topic's 4
relevant docs share fewer distinctive keywords and fall outside the top-10 under the
bag-of-words embedder — this is a property of the deliberately simple stub embedder,
not an index defect.

## Interpretation

- **Recall / speed trade-off holds as expected.** Flat is exact but its per-query cost
  grows linearly (throughput 1,970 → 75 q/s from 1k → 10k). LSH is the fastest per
  query and cheapest to build, at the lowest recall (~0.80). IVF gives a strong
  recall/latency balance (0.925–0.990) with moderate build cost.
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
