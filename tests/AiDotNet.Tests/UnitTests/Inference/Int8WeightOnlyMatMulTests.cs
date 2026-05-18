using AiDotNet.Inference.Quantization;
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
        var rng = new Random(0xC0DE);

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
