using AiDotNet.Inference.Quantization;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Inference;

/// <summary>
/// Regression test for AiDotNet#1363 — the scalar dequant-on-fly matmul in
/// <c>QuantizedDenseLayer</c> / <c>QuantizedAttentionLayer</c> was replaced
/// with a tiled SGEMM + AVX2 dequant path through <c>Int8WeightOnlyMatMul</c>.
/// These tests verify the new helper agrees with the original scalar
/// reference at FP32 precision (≤ 1 ULP per element typical, well below the
/// per-row symmetric int8 quantization noise floor).
/// </summary>
public class Int8WeightOnlyMatMulTests
{
    private static void ScalarReference(
        ReadOnlySpan<float> input,
        sbyte[] weightsInt8,
        float[] rowScales,
        float[]? biases,
        Span<float> output,
        int rows,
        int inputSize,
        int outputSize)
    {
        for (int r = 0; r < rows; r++)
        {
            int inputBase = r * inputSize;
            int outputBase = r * outputSize;
            for (int o = 0; o < outputSize; o++)
            {
                float sum = biases != null ? biases[o] : 0f;
                float scale = rowScales[o];
                int wBase = o * inputSize;
                for (int i = 0; i < inputSize; i++)
                {
                    sum += input[inputBase + i] * (weightsInt8[wBase + i] * scale);
                }
                output[outputBase + o] = sum;
            }
        }
    }

    [Theory]
    [InlineData(1, 16, 16, true)]    // smaller-than-tile
    [InlineData(4, 32, 64, true)]    // normal small
    [InlineData(8, 128, 256, true)]  // multi-tile typical transformer FFN
    [InlineData(2, 512, 2048, true)] // BERT FFN scale
    [InlineData(3, 33, 47, true)]    // unaligned shapes
    [InlineData(4, 128, 64, false)]  // no bias
    public void MultiplyAddBias_MatchesScalarReference(
        int rows, int inputSize, int outputSize, bool withBias)
    {
        var rng = RandomHelper.CreateSeededRandom(0xC0DE); // CSPRNG-aware seeded Random per RandomHelper contract; deterministic seed retained for test reproducibility

        var input = new float[rows * inputSize];
        for (int i = 0; i < input.Length; i++)
            input[i] = (float)(rng.NextDouble() * 2 - 1);

        var weightsInt8 = new sbyte[outputSize * inputSize];
        for (int i = 0; i < weightsInt8.Length; i++)
            weightsInt8[i] = (sbyte)rng.Next(-127, 128);

        var rowScales = new float[outputSize];
        for (int o = 0; o < outputSize; o++)
            rowScales[o] = (float)(rng.NextDouble() * 0.01 + 0.001);

        float[]? biases = null;
        if (withBias)
        {
            biases = new float[outputSize];
            for (int o = 0; o < outputSize; o++)
                biases[o] = (float)(rng.NextDouble() * 0.1);
        }

        var expected = new float[rows * outputSize];
        ScalarReference(input, weightsInt8, rowScales, biases, expected,
            rows, inputSize, outputSize);

        var actual = new float[rows * outputSize];
        Int8WeightOnlyMatMul.MultiplyAddBias(
            input, weightsInt8, rowScales, biases, actual,
            rows, inputSize, outputSize);

        // The SIMD path and the scalar reference compute the same mathematical
        // expression; the only divergence is float-summation order (BLIS k-loop
        // packing vs naive sequential). Allow a tolerance proportional to the
        // accumulation magnitude — ≈ 2 × max|value| × √K × 1e-6 (single ULP at
        // each FMA, accumulated in sqrt-mean over independent paths).
        double maxAbs = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double a = Math.Abs(expected[i]);
            if (a > maxAbs) maxAbs = a;
        }
        double absTol = 2.0 * Math.Max(1e-3, maxAbs) * Math.Sqrt(inputSize) * 1.5e-6;

        for (int i = 0; i < expected.Length; i++)
        {
            double diff = Math.Abs(expected[i] - actual[i]);
            Assert.True(
                diff <= absTol,
                $"Element {i}: expected {expected[i]}, got {actual[i]}, diff={diff}, absTol={absTol}, maxAbs={maxAbs}");
        }
    }

    [Fact]
    public void MultiplyAddBias_NoBias_ClearsOutputBetweenTiles()
    {
        // Single-row sanity check: with all weights = 0 and no bias, the
        // tile-output scatter must not leak prior contents into the destination.
        int rows = 1, inputSize = 64, outputSize = 256;

        var input = new float[rows * inputSize];
        for (int i = 0; i < input.Length; i++) input[i] = 1.0f;

        var weightsInt8 = new sbyte[outputSize * inputSize]; // all zeros
        var rowScales = new float[outputSize];
        for (int o = 0; o < outputSize; o++) rowScales[o] = 0.123f;

        // Pre-fill output with sentinel so we can detect any leak from
        // ArrayPool-rented tile scratch into the user-visible buffer.
        var output = new float[rows * outputSize];
        for (int i = 0; i < output.Length; i++) output[i] = 999.0f;

        Int8WeightOnlyMatMul.MultiplyAddBias(
            input, weightsInt8, rowScales, biases: null, output,
            rows, inputSize, outputSize);

        for (int i = 0; i < output.Length; i++)
        {
            Assert.True(output[i] == 0f,
                $"Expected 0 at index {i}, got {output[i]} — tile scratch leaked into output.");
        }
    }

    [Fact]
    public void MultiplyAddBias_ZeroRows_DoesNotThrow()
    {
        var input = Array.Empty<float>();
        var weights = new sbyte[16];
        var scales = new float[4];
        var output = Array.Empty<float>();

        Int8WeightOnlyMatMul.MultiplyAddBias(
            input, weights, scales, biases: null, output,
            rows: 0, inputSize: 4, outputSize: 4);
        // No assertion needed — the contract is "do not throw / segfault".
    }

    [Fact]
    public void MultiplyAddBias_ZeroRows_OversizedOutputBuffer_PreservesSentinels()
    {
        // The early-return for rows == 0 must NOT zero the caller's span when
        // the span is larger than the logical (rows*outputSize == 0) region.
        // Pre-fill an oversized buffer with sentinels and assert they survive.
        var input = Array.Empty<float>();
        var weights = new sbyte[16];
        var scales = new float[4];
        var output = new float[8];
        for (int i = 0; i < output.Length; i++) output[i] = 12345.0f;

        Int8WeightOnlyMatMul.MultiplyAddBias(
            input, weights, scales, biases: null, output.AsSpan(),
            rows: 0, inputSize: 4, outputSize: 4);

        for (int i = 0; i < output.Length; i++)
        {
            Assert.Equal(12345.0f, output[i]);
        }
    }

    [Fact]
    public void MultiplyAddBias_OutputSizeZero_OversizedOutputBuffer_PreservesSentinels()
    {
        // Same as above but for the outputSize == 0 early-return branch.
        var input = new float[8];
        var weights = Array.Empty<sbyte>();
        var scales = Array.Empty<float>();
        var output = new float[8];
        for (int i = 0; i < output.Length; i++) output[i] = 54321.0f;

        Int8WeightOnlyMatMul.MultiplyAddBias(
            input, weights, scales, biases: null, output.AsSpan(),
            rows: 2, inputSize: 4, outputSize: 0);

        for (int i = 0; i < output.Length; i++)
        {
            Assert.Equal(54321.0f, output[i]);
        }
    }

    [Fact]
    public void MultiplyAddBias_InputSizeZero_ThrowsArgumentOutOfRange()
    {
        // The inputSize == 0 case is now rejected explicitly (review #1363
        // C6XGR) — production callers validate input dim upstream and
        // a positive rows/outputSize combined with inputSize=0 only arrives
        // from a misconfigured caller. Failing fast surfaces the upstream
        // bug at the matmul call instead of producing a silent
        // zero-or-bias-only result.
        var input = Array.Empty<float>();
        var weights = Array.Empty<sbyte>();
        var scales = new[] { 0.1f, 0.2f, 0.3f };
        var biases = new[] { 1.5f, 2.5f, 3.5f };
        var output = new float[6]; // 2 rows * 3 outputs

        var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
        {
            Int8WeightOnlyMatMul.MultiplyAddBias(
                input, weights, scales, biases, output.AsSpan(),
                rows: 2, inputSize: 0, outputSize: 3);
        });
        Assert.Equal("inputSize", ex.ParamName);
        Assert.Contains("must be positive", ex.Message);
    }

    [Fact]
    public void MultiplyAddBias_MultiTile_WithBias_BiasAppliedInEveryTile()
    {
        // Exercises the `biases != null` branch in the MULTI-tile path
        // (review #1363 C6XHK — the prior MultiTile test had biases=null
        // and the prior MatchesScalarReference cases used inputSize<=512
        // where ChooseTileSize returned outputSize, so the multi-tile
        // bias-add scatter loop was never tested in isolation).
        //
        // inputSize=8192 forces ChooseTileSize ≈ 16; with outputSize=64
        // that's 4 tiles. Bias is row-shaped (length outputSize) so the
        // scatter must offset by oBase on each tile and produce the
        // correct sum per output column.
        const int rows = 2;
        const int inputSize = 8192;
        const int outputSize = 64;

        // Deterministic small weights and biases so we can compute the
        // expected output exactly without an SIMD-equivalence reference.
        var input = new float[rows * inputSize];
        for (int i = 0; i < input.Length; i++) input[i] = 1.0f; // every input element = 1

        var weights = new sbyte[outputSize * inputSize];
        for (int i = 0; i < weights.Length; i++) weights[i] = 1; // every weight = 1

        var scales = new float[outputSize];
        for (int o = 0; o < outputSize; o++) scales[o] = 0.5f; // every scale = 0.5

        var biases = new float[outputSize];
        for (int o = 0; o < outputSize; o++) biases[o] = (float)o * 0.25f; // distinct per-output-column bias (length == outputSize; indexed by output column o in [0, outputSize), NOT by row — review #1363 C8QYR)

        var output = new float[rows * outputSize];

        Int8WeightOnlyMatMul.MultiplyAddBias(
            input, weights, scales, biases, output.AsSpan(),
            rows: rows, inputSize: inputSize, outputSize: outputSize);

        // Each output cell = sum_k(input[r,k] * (weights[o,k]*scale[o])) + bias[o]
        //                 = inputSize * (1 * 0.5) + 0.25 * o
        //                 = 4096 + 0.25 * o
        // Note: SimdGemm uses fp32 arithmetic — for inputSize=8192 the
        // exact result is well within float precision.
        float baseDot = inputSize * 0.5f; // 4096
        for (int r = 0; r < rows; r++)
        {
            for (int o = 0; o < outputSize; o++)
            {
                float expected = baseDot + biases[o];
                Assert.True(
                    Math.Abs(output[r * outputSize + o] - expected) < 1e-2f,
                    $"At [{r},{o}] expected ≈{expected}, got {output[r * outputSize + o]}. " +
                    $"Multi-tile bias scatter offset by oBase per tile must apply biases[{o}]={biases[o]}.");
            }
        }
    }

    [Fact]
    public void MultiplyAddBias_MultiTile_ActuallyTilesOutput()
    {
        // Force an outputSize that exceeds ChooseTileSize so the multi-tile
        // scatter path is exercised — the existing "scatter doesn't leak" test
        // happens to choose dimensions where ChooseTileSize returns outputSize
        // and only one tile runs. inputSize=8192 → ChooseTileSize ≈ 16; with
        // outputSize=64 we get 4 tiles.
        //
        // Test-fixture invariant: this test EXPLICITLY depends on
        // ChooseTileSize's current heuristic returning a value < outputSize
        // for (outputSize=64, inputSize=8192). If a future tuning of
        // ChooseTileSize raises the multi-tile threshold past this point,
        // the assert below will fire with a clear "test fixture invariant"
        // message — the failing test then signals "pick a larger
        // outputSize / inputSize to force tiling" rather than the test
        // silently degrading to a single-tile path (review #1363 C8QYu).
        const int rows = 2;
        const int inputSize = 8192;
        const int outputSize = 64;

        int chosen = Int8WeightOnlyMatMul.ChooseTileSize(outputSize, inputSize);
        Assert.True(chosen < outputSize,
            $"Test fixture invariant broken: ChooseTileSize({outputSize}, {inputSize}) = {chosen} " +
            $"must be < outputSize so the multi-tile scatter path runs. Either ChooseTileSize's " +
            $"tile-selection heuristic changed (raise outputSize / inputSize in this test to force " +
            $"tiling), or the heuristic is intentionally collapsing single-tile for these dims " +
            $"(then split this test into a separate fixture that hits the multi-tile path).");

        var rng = RandomHelper.CreateSeededRandom(0xC0DE); // CSPRNG-aware seeded Random per RandomHelper contract; deterministic seed retained for test reproducibility
        var input = new float[rows * inputSize];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
        var weights = new sbyte[outputSize * inputSize];
        for (int i = 0; i < weights.Length; i++) weights[i] = (sbyte)rng.Next(-127, 128);
        var scales = new float[outputSize];
        for (int o = 0; o < outputSize; o++) scales[o] = (float)(rng.NextDouble() * 0.01 + 0.001);

        var output = new float[rows * outputSize];
        // Pre-fill with sentinel so an unwritten tile would surface.
        for (int i = 0; i < output.Length; i++) output[i] = 9999.0f;

        Int8WeightOnlyMatMul.MultiplyAddBias(
            input, weights, scales, biases: null, output,
            rows, inputSize, outputSize);

        // Every output element must have been overwritten — sentinel cannot survive.
        for (int i = 0; i < output.Length; i++)
        {
            Assert.NotEqual(9999.0f, output[i]);
        }
    }

    [Theory]
    [InlineData(16, 16)]
    [InlineData(64, 4096)]
    [InlineData(128, 1)]
    [InlineData(8192, 32)]
    public void ChooseTileSize_StaysWithinBounds(int outputSize, int inputSize)
    {
        int tile = Int8WeightOnlyMatMul.ChooseTileSize(outputSize, inputSize);
        Assert.True(tile > 0, "Tile size must be positive.");
        Assert.True(tile <= outputSize, $"Tile {tile} must not exceed outputSize {outputSize}.");
        // Either we returned at most outputSize (tiny case) or a 16-aligned tile.
        Assert.True(tile == outputSize || tile % 16 == 0,
            $"Tile {tile} must be a multiple of 16 unless it equals outputSize.");
    }
}
