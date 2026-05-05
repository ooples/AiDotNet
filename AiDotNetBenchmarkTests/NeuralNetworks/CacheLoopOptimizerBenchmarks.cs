using System;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Attention;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Optimization;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.NeuralNetworks;

// Issue #1158 verification harness — measures whether
// LoopOptimizer / CacheOptimizer retrofits actually move the needle on
// the four hot-path categories the issue calls out:
//   1. FlashAttention block iteration
//   2. LinearAlgebra matmul fallback
//   3. Loss-function gradients
//   4. Embedding scatter / gather
// Each Bench class has a [Naive] + [Optimized] variant on identical input
// so the diff tells us whether to ship the retrofit. This is the
// gating evidence for the PR — categories that don't show measurable
// speedup get documented as null-results, not silently retrofitted.

/// <summary>
/// FlashAttention forward pass: compares fixed-default block sizes (64 / 64
/// from FlashAttentionConfig.Default) against cache-aware block sizes from
/// LoopOptimizer.DetermineOptimalTileSize. The expectation is that on
/// systems where the L1 cache differs significantly from the assumption
/// behind 64×64, the cache-aware tile picker reduces capacity misses.
/// </summary>
[SimpleJob(RuntimeMoniker.Net80, warmupCount: 3, iterationCount: 5)]
[MemoryDiagnoser]
public class FlashAttentionTilingBenchmark
{
    [Params(128, 512, 1024)]
    public int SeqLen { get; set; }

    [Params(64)]
    public int HeadDim { get; set; }

    private Tensor<float> _q = null!;
    private Tensor<float> _k = null!;
    private Tensor<float> _v = null!;
    private FlashAttentionConfig _defaultConfig = null!;
    private FlashAttentionConfig _cacheAwareConfig = null!;

    [GlobalSetup]
    public void Setup()
    {
        _q = MakeRandom(new[] { 1, SeqLen, HeadDim });
        _k = MakeRandom(new[] { 1, SeqLen, HeadDim });
        _v = MakeRandom(new[] { 1, SeqLen, HeadDim });

        _defaultConfig = FlashAttentionConfig.Default;

        // Cache-aware sizing: pick block sizes from L1, capped at SeqLen and at
        // the float element size (4 bytes). Matches the retrofit candidate.
        int tile = LoopOptimizer.DetermineOptimalTileSize(SeqLen, sizeof(float));
        _cacheAwareConfig = new FlashAttentionConfig
        {
            BlockSizeQ = tile,
            BlockSizeKV = tile,
            UseGpuKernel = false, // Force the CPU path so the benchmark measures what we changed.
        };
    }

    [Benchmark(Baseline = true, Description = "Hardcoded BlockSize=64")]
    public Tensor<float> Naive_Default64()
    {
        var (output, _) = FlashAttention<float>.Forward(_q, _k, _v, _defaultConfig);
        return output;
    }

    [Benchmark(Description = "DetermineOptimalTileSize from L1")]
    public Tensor<float> Optimized_CacheAware()
    {
        var (output, _) = FlashAttention<float>.Forward(_q, _k, _v, _cacheAwareConfig);
        return output;
    }

    private static Tensor<float> MakeRandom(int[] shape)
    {
        var t = new Tensor<float>(shape);
        // Deterministic PRNG so benchmark runs are comparable across invocations.
        var rng = new Random(42);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5);
        return t;
    }
}

/// <summary>
/// Loss-gradient inner loop: compares a hand-rolled element-wise gradient
/// (the current pattern in BinaryCrossEntropyLoss / similar) against
/// LoopOptimizer.UnrollBy4. The closure capture in UnrollBy4 boxes any
/// captured local — for accumulation-style loops this defeats register
/// optimization, so the optimized variant is expected to be SLOWER. The
/// benchmark exists to confirm that conclusion empirically before we
/// document it as a null-result in the issue write-up.
/// </summary>
[SimpleJob(RuntimeMoniker.Net80, warmupCount: 3, iterationCount: 5)]
[MemoryDiagnoser]
public class LossGradientUnrollBenchmark
{
    [Params(1024, 8192, 65536)]
    public int Length { get; set; }

    private Vector<double> _predicted = null!;
    private Vector<double> _actual = null!;
    private BinaryCrossEntropyLoss<double> _loss = null!;

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);
        _predicted = new Vector<double>(Length);
        _actual = new Vector<double>(Length);
        for (int i = 0; i < Length; i++)
        {
            _predicted[i] = 0.05 + rng.NextDouble() * 0.9;  // probability in [0.05, 0.95]
            _actual[i] = rng.NextDouble() < 0.5 ? 0.0 : 1.0;
        }
        _loss = new BinaryCrossEntropyLoss<double>();
    }

    [Benchmark(Baseline = true, Description = "Naive elementwise loop (current)")]
    public Vector<double> Naive_BCEDerivative() => _loss.CalculateDerivative(_predicted, _actual);

    [Benchmark(Description = "UnrollBy4 with closure capture")]
    public Vector<double> Optimized_UnrollBy4()
    {
        // Mirrors BinaryCrossEntropyLoss.CalculateDerivative pattern but routes
        // the inner index loop through LoopOptimizer.UnrollBy4. The captured
        // locals (predicted, actual, result) are required so the lambda closes
        // over them — this IS the boxing/closure cost we're measuring.
        var p = _predicted;
        var a = _actual;
        int n = p.Length;
        var result = new Vector<double>(n);
        double scale = 1.0 / n;
        LoopOptimizer.UnrollBy4(n, i =>
        {
            // -[(actual / predicted) - ((1-actual)/(1-predicted))] / n
            double pi = Math.Max(1e-12, Math.Min(1 - 1e-12, p[i]));
            double ai = a[i];
            result[i] = -((ai / pi) - ((1 - ai) / (1 - pi))) * scale;
        });
        return result;
    }
}

/// <summary>
/// Embedding lookup scatter: compares the current row-major double-loop
/// (LookupTokens-style) against LoopOptimizer.Tile2D over (token, dim).
/// The current loop is already row-major contiguous — the embedding row at
/// _embeddingTensor[tokenId, *] is laid out in memory exactly as it's
/// scanned — so Tile2D doesn't reorder anything beneficial; it just adds
/// a closure call per tile. Benchmark exists to confirm the no-win
/// hypothesis before documenting it.
/// </summary>
[SimpleJob(RuntimeMoniker.Net80, warmupCount: 3, iterationCount: 5)]
[MemoryDiagnoser]
public class EmbeddingScatterBenchmark
{
    [Params(64, 512)]
    public int TokenCount { get; set; }

    [Params(128, 768)]
    public int EmbeddingDim { get; set; }

    private const int VocabSize = 4096;
    private double[,] _embeddings = null!;
    private int[] _tokenIds = null!;

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);
        _embeddings = new double[VocabSize, EmbeddingDim];
        for (int v = 0; v < VocabSize; v++)
            for (int d = 0; d < EmbeddingDim; d++)
                _embeddings[v, d] = rng.NextDouble();

        _tokenIds = new int[TokenCount];
        for (int i = 0; i < TokenCount; i++) _tokenIds[i] = rng.Next(VocabSize);
    }

    [Benchmark(Baseline = true, Description = "Naive nested for loops (current pattern)")]
    public double[,] Naive_Lookup()
    {
        var result = new double[TokenCount, EmbeddingDim];
        for (int i = 0; i < TokenCount; i++)
        {
            int tokenId = _tokenIds[i];
            for (int d = 0; d < EmbeddingDim; d++)
            {
                result[i, d] = _embeddings[tokenId, d];
            }
        }
        return result;
    }

    [Benchmark(Description = "Tile2D + per-tile inner copy")]
    public double[,] Optimized_Tile2D()
    {
        var result = new double[TokenCount, EmbeddingDim];
        int tile = LoopOptimizer.DetermineOptimalTileSize(EmbeddingDim, sizeof(double));
        // Tile2D over (token, dim) with the inner block doing a contiguous copy
        // along d — same access pattern as Naive but with a tile barrier in the
        // outer dimension.
        LoopOptimizer.Tile2D(TokenCount, EmbeddingDim, tile, (i0, i1, j0, j1) =>
        {
            for (int i = i0; i < i1; i++)
            {
                int tokenId = _tokenIds[i];
                for (int d = j0; d < j1; d++)
                {
                    result[i, d] = _embeddings[tokenId, d];
                }
            }
        });
        return result;
    }
}

/// <summary>
/// Naive matmul vs ComputeOptimalTiling-driven blocked matmul. The issue's
/// audit hint mentions "remaining hand-rolled matmul fallbacks" in
/// AiDotNet's LinearAlgebra. Matmul has actually moved to the Tensors
/// package (where Engine.MatrixMultiply is already cache-aware), so this
/// benchmark stands in for the hypothetical fallback rather than measuring
/// real production code. Useful for confirming that ComputeOptimalTiling
/// genuinely beats a naive triple-loop on representative shapes; the
/// resulting +ratio data informs whether other matmul-shaped kernels
/// (custom reductions, etc.) would benefit from the same retrofit.
/// </summary>
[SimpleJob(RuntimeMoniker.Net80, warmupCount: 3, iterationCount: 5)]
[MemoryDiagnoser]
public class MatmulTilingBenchmark
{
    [Params(128, 256, 512)]
    public int Size { get; set; }

    private float[] _a = null!;
    private float[] _b = null!;

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);
        _a = new float[Size * Size];
        _b = new float[Size * Size];
        for (int i = 0; i < _a.Length; i++) _a[i] = (float)rng.NextDouble();
        for (int i = 0; i < _b.Length; i++) _b[i] = (float)rng.NextDouble();
    }

    [Benchmark(Baseline = true, Description = "Naive ijk triple-loop")]
    public float[] Naive_Matmul()
    {
        int n = Size;
        var c = new float[n * n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
            {
                float acc = 0;
                for (int k = 0; k < n; k++)
                    acc += _a[i * n + k] * _b[k * n + j];
                c[i * n + j] = acc;
            }
        return c;
    }

    [Benchmark(Description = "ComputeOptimalTiling-driven blocked matmul")]
    public float[] Optimized_BlockedMatmul()
    {
        int n = Size;
        var c = new float[n * n];
        var (tileM, tileN, tileK) = CacheOptimizer.ComputeOptimalTiling(n, n, n, sizeof(float));
        // Standard ikj-blocked matmul with cache-aware tile picker.
        for (int i0 = 0; i0 < n; i0 += tileM)
        {
            int iEnd = Math.Min(i0 + tileM, n);
            for (int k0 = 0; k0 < n; k0 += tileK)
            {
                int kEnd = Math.Min(k0 + tileK, n);
                for (int j0 = 0; j0 < n; j0 += tileN)
                {
                    int jEnd = Math.Min(j0 + tileN, n);
                    for (int i = i0; i < iEnd; i++)
                        for (int k = k0; k < kEnd; k++)
                        {
                            float aik = _a[i * n + k];
                            for (int j = j0; j < jEnd; j++)
                                c[i * n + j] += aik * _b[k * n + j];
                        }
                }
            }
        }
        return c;
    }
}
