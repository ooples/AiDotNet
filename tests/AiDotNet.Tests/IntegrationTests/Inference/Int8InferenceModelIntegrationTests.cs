using System;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Inference.Quantization;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.Inference;

/// <summary>
/// Integration tests for the public <see cref="Int8InferenceModel"/> surface introduced in
/// AiDotNet#1342. Verifies that quantizing a trained float network produces:
///   1. Quality: INT8 output stays within tolerance of the FP32 baseline.
///   2. Storage: INT8 weight bytes are ~4x smaller than FP32 weight bytes.
///   3. Surface: <c>Predict</c> runs end-to-end without exposing internal sealed types.
/// </summary>
public class Int8InferenceModelIntegrationTests
{
    private readonly ITestOutputHelper _output;

    public Int8InferenceModelIntegrationTests(ITestOutputHelper output)
    {
        _output = output;
    }

    private static Tensor<float> CreateRandomInput(int seqLen, int embDim, int seed = 1342)
    {
        // Transformer<float> expects [batch, seqLen, embDim] for its attention path; build a
        // unit-batch tensor so callers can stay shape-agnostic about the batch axis.
        var rng = new Random(seed);
        var data = new float[seqLen * embDim];
        for (int i = 0; i < data.Length; i++)
            data[i] = (float)(rng.NextDouble() * 2 - 1);
        return new Tensor<float>(data, new[] { 1, seqLen, embDim });
    }

    /// <summary>
    /// Builds a small Transformer&lt;float&gt; with concrete weights and runs a single warm-up
    /// predict so the lazy attention layers shape-resolve before quantization. Returns the
    /// warmed model.
    /// </summary>
    private static Transformer<float> BuildAndWarmTransformer(
        int seqLen, int embDim, int numHeads, int seed = 1342)
    {
        var architecture = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            numEncoderLayers: 1,
            numDecoderLayers: 0,
            numHeads: numHeads,
            modelDimension: embDim,
            feedForwardDimension: embDim * 2,
            inputSize: embDim,
            outputSize: embDim,
            maxSequenceLength: seqLen);

        var net = new Transformer<float>(architecture);

        // Warm-up predict so lazy MHA layers materialise their Q/K/V/O weights.
        var warm = CreateRandomInput(seqLen, embDim, seed);
        net.Predict(warm);
        return net;
    }

    [Fact(Timeout = 120000)]
    public async Task FromTrained_ReturnsModelThatPredictsWithoutErrors()
    {
        await Task.Yield();
        int seqLen = 8;
        int embDim = 32;
        int numHeads = 4;

        var fp32 = BuildAndWarmTransformer(seqLen, embDim, numHeads);
        var input = CreateRandomInput(seqLen, embDim, seed: 11);

        var int8 = Int8InferenceModel.FromTrained(fp32);
        var output = int8.Predict(input);

        Assert.NotNull(output);
        Assert.True(output.Length > 0, "INT8 predict should produce a non-empty output tensor.");
        Assert.True(int8.QuantizedLayerCount > 0,
            "At least one layer (MHA or Dense) should have been quantized.");
    }

    [Fact(Timeout = 120000)]
    public async Task FromTrained_QuantizedOutputIsWithinToleranceOfFP32()
    {
        await Task.Yield();
        int seqLen = 8;
        int embDim = 32;
        int numHeads = 4;

        var fp32 = BuildAndWarmTransformer(seqLen, embDim, numHeads, seed: 23);
        var input = CreateRandomInput(seqLen, embDim, seed: 23);

        // Capture FP32 baseline first.
        var fp32Output = fp32.Predict(input);

        // Quantize a CLONE so the FP32 baseline above is untouched.
        var int8 = Int8InferenceModel.FromTrained(fp32, cloneModel: true);
        var int8Output = int8.Predict(input);

        Assert.Equal(fp32Output.Shape.ToArray(), int8Output.Shape.ToArray());

        // Element-wise relative error. For per-row symmetric INT8 with random-uniform inputs,
        // the typical SNR is 25-40 dB which translates to <5% mean relative error.
        double sumSqErr = 0;
        double sumSqSig = 0;
        for (int i = 0; i < fp32Output.Length; i++)
        {
            double err = int8Output[i] - fp32Output[i];
            sumSqErr += err * err;
            sumSqSig += (double)fp32Output[i] * fp32Output[i];
        }
        double snrDb = sumSqErr > 0
            ? 10.0 * Math.Log10(sumSqSig / Math.Max(sumSqErr, 1e-30))
            : double.PositiveInfinity;

        // 10 dB is a conservative floor for INT8 weight-only on small networks where
        // quantization-noise accumulates across two layers (MHA Q/K/V/O + FFN Dense). Real
        // transformer inference typically clears 25 dB on this kind of canary.
        Assert.True(snrDb > 10.0,
            $"INT8 vs FP32 signal-to-noise ratio {snrDb:F1} dB is below the 10 dB floor.");
    }

    [Fact(Timeout = 120000)]
    public async Task FromTrained_ReportsCompressionRatioApproachingFourX()
    {
        await Task.Yield();
        // Use a larger embDim so the per-row-scale overhead (4 bytes per output row) shrinks
        // relative to the per-weight 1-byte savings, pushing the ratio closer to 4.0.
        int seqLen = 8;
        int embDim = 128;
        int numHeads = 8;

        var fp32 = BuildAndWarmTransformer(seqLen, embDim, numHeads, seed: 47);
        var int8 = Int8InferenceModel.FromTrained(fp32);

        Assert.True(int8.QuantizedWeightBytes > 0, "Quantized weight bytes must be non-zero.");
        Assert.True(int8.OriginalWeightBytes > 0, "Original weight bytes must be non-zero.");

        // Asymptotic ratio for per-row symmetric INT8:
        //   FP32 storage = numWeights * 4 + biasBytes
        //   INT8 storage = numWeights * 1 + outRows * 4 + biasBytes
        // For embDim=128 with MHA + FFN dense layers, the achieved ratio should land between
        // 3.5x and 4.0x. We require >3.0x to leave headroom for the FP32 bias terms.
        Assert.True(int8.CompressionRatio > 3.0,
            $"Compression ratio {int8.CompressionRatio:F2}x is below 3.0x floor. " +
            $"Quantized: {int8.QuantizedWeightBytes} bytes, original: {int8.OriginalWeightBytes} bytes.");
    }

    [Fact(Timeout = 180000)]
    public async Task FromTrained_PredictWallClockGapVsFP32_DocumentsCurrentState()
    {
        // Wall-clock perf gap for AiDotNet#1342. The SIMD INT8 wiring landed
        // (#1363 / Int8WeightOnlyMatMul.MultiplyAddBias) — QuantizedAttentionLayer
        // and QuantizedDenseLayer now route through AiDotNet.Tensors' tiled SGEMM
        // + AVX2 dequant primitives instead of the scalar 3-loop dequant-on-fly
        // matmul this test originally documented.
        //
        // Measured baseline on this 16x64 canary post-SIMD: ratio ≈ 15x
        // (int8 ~5.4 ms vs fp32 ~0.35 ms). The gap is no longer the scalar inner
        // loop — at this canary size the GEMM body (16×64×64 = 65k FMAs) is
        // small enough that per-call overhead dominates: per-row dequant calls,
        // ArrayPool.Rent/Return, scatter into the strided output. The FP32 path
        // goes through a single FusedLinear GEMM with none of that bookkeeping.
        //
        // Tightening below ~20x at THIS canary requires a Tensors-side primitive
        // that keeps weights as sbyte all the way through the kernel (true 4x
        // DRAM bandwidth saving on weight loads plus elimination of the per-tile
        // dequant scratch). That work is tracked separately. On BERT-class
        // shapes (e.g. 768×3072) the GEMM body dominates and the same Int8-
        // WeightOnlyMatMul wiring lands much closer to FP32 — but the
        // diagnostic-time-budgeted canary fixture is what's measured here.
        //
        // The ceiling tightens from the pre-SIMD 50x guard to 20x: tight enough
        // to catch a real regression, loose enough to absorb CI variance on
        // warm/cold cache and turbo-boost noise at the small canary size.
        await Task.Yield();
        int seqLen = 16;
        int embDim = 64;
        int numHeads = 4;
        const int warmupIters = 5;
        const int measureIters = 20;

        var fp32 = BuildAndWarmTransformer(seqLen, embDim, numHeads, seed: 53);
        // Explicit cloneModel: true so FromTrained's INT8 rewrite
        // operates on a deep copy of fp32 and leaves the original
        // intact for the FP32-baseline measurement below. The
        // default *is* true, but in a perf test that compares
        // fp32 vs int8 on the same machine relying on an implicit
        // default is fragile — a future signature change that
        // flipped the default to false would silently turn this
        // into a quantized-vs-quantized comparison and the <50x
        // guard would lose all signal. (PR #1348 review comment.)
        var int8 = Int8InferenceModel.FromTrained(fp32, cloneModel: true);
        var input = CreateRandomInput(seqLen, embDim, seed: 31);

        // Warmup both paths to amortize JIT, plan cache, etc.
        for (int i = 0; i < warmupIters; i++)
        {
            fp32.Predict(input);
            int8.Predict(input);
        }

        var swFp32 = Stopwatch.StartNew();
        for (int i = 0; i < measureIters; i++)
            fp32.Predict(input);
        swFp32.Stop();

        var swInt8 = Stopwatch.StartNew();
        for (int i = 0; i < measureIters; i++)
            int8.Predict(input);
        swInt8.Stop();

        double fp32Ms = swFp32.Elapsed.TotalMilliseconds / measureIters;
        double int8Ms = swInt8.Elapsed.TotalMilliseconds / measureIters;
        double ratio = int8Ms / Math.Max(fp32Ms, 0.0001);

        _output.WriteLine(
            $"INT8 wall-clock ratio: {ratio:F2}x (int8={int8Ms:F3}ms vs fp32={fp32Ms:F3}ms) " +
            $"on {seqLen}x{embDim} transformer canary, {numHeads} heads, {measureIters} iters.");

        // 20x post-SIMD ceiling (down from the pre-SIMD 50x). The measured
        // canary ratio is ~15x — small-matrix per-call overhead dominates here.
        // A larger-shape benchmark would show much closer to FP32; that is what
        // a future int8-through-the-kernel GEMM would unlock at canary sizes too.
        Assert.True(ratio < 20.0,
            $"INT8 wall-clock ratio {ratio:F2}x (int8={int8Ms:F3}ms vs fp32={fp32Ms:F3}ms) " +
            $"exceeds the 20x post-SIMD regression ceiling. The Int8WeightOnlyMatMul.MultiplyAddBias " +
            $"tiled SGEMM + AVX2 dequant path keeps this around 15x on the canary; a number above 20x " +
            $"indicates a regression in the dequant primitive, the tile size choice, or the engine " +
            $"FusedLinear baseline. Run the test under a profiler if you see this.");
    }

    [Fact(Timeout = 60000)]
    public async Task FromTrained_RejectsNullModel()
    {
        await Task.Yield();
        Assert.Throws<ArgumentNullException>(() => Int8InferenceModel.FromTrained(null!));
    }

    [Fact(Timeout = 120000)]
    public async Task FromTrained_CloneModeLeavesOriginalUntouched()
    {
        await Task.Yield();
        int seqLen = 8;
        int embDim = 32;
        int numHeads = 4;

        var fp32 = BuildAndWarmTransformer(seqLen, embDim, numHeads, seed: 67);
        int originalLayerCount = fp32.Layers.Count;
        var originalLayerTypes = fp32.Layers.Select(l => l.GetType()).ToArray();

        var int8 = Int8InferenceModel.FromTrained(fp32, cloneModel: true);

        // Original model must still have FP32 attention / dense layers.
        Assert.Equal(originalLayerCount, fp32.Layers.Count);
        for (int i = 0; i < originalLayerCount; i++)
        {
            Assert.Equal(originalLayerTypes[i], fp32.Layers[i].GetType());
        }

        // But the wrapper's inner model must have at least one rewritten layer.
        Assert.True(int8.QuantizedLayerCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task FromTrained_StatsReflectActualByteCounts()
    {
        await Task.Yield();
        int seqLen = 8;
        int embDim = 32;
        int numHeads = 4;

        var fp32 = BuildAndWarmTransformer(seqLen, embDim, numHeads, seed: 71);
        var int8 = Int8InferenceModel.FromTrained(fp32);

        Assert.True(int8.OriginalWeightBytes > int8.QuantizedWeightBytes,
            "Original weight bytes must exceed quantized weight bytes for any non-trivial network.");
        // Even small embDim should beat 2x with one MHA + at least one FFN dense layer.
        Assert.True(int8.CompressionRatio > 2.0,
            $"Even small embDim should beat 2x. Got {int8.CompressionRatio:F2}x.");
    }
}
