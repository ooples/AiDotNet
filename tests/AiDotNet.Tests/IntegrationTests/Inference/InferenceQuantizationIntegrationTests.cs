using System;
using System.Linq;
using AiDotNet.Deployment.Optimization.Quantization.Formats;
using AiDotNet.Inference.Quantization;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Inference;

/// <summary>
/// Deep mathematical correctness tests for INT8 and FP8 quantization.
/// Verifies quantize/dequantize round-trip accuracy, scale computation,
/// clamping behavior, and FP8 E4M3/E5M2 bit-exact encoding/decoding.
/// </summary>
public class InferenceQuantizationIntegrationTests
{
    private const double Tolerance = 1e-4;

    #region INT8 Quantization - Scale Computation Golden References

    [Fact]
    public void Int8_PerRowScale_GoldenReference()
    {
        // Row [0.5, -1.0, 0.25, 0.75]: maxAbs = 1.0
        // scale = maxAbs / 127 = 1.0 / 127 ≈ 0.007874
        // inv = 127.0
        // quantized: [round(0.5*127), round(-1.0*127), round(0.25*127), round(0.75*127)]
        //          = [64, -127, 32, 95]
        float[] data = { 0.5f, -1.0f, 0.25f, 0.75f };
        var result = Int8WeightOnlyQuantization.QuantizePerRow(data.AsSpan(), rows: 1, cols: 4);

        float expectedScale = 1.0f / 127f;
        Assert.Equal(expectedScale, result.Scales[0], 1e-6);
        Assert.Equal(64, result.Weights[0]);
        Assert.Equal(-127, result.Weights[1]);
        Assert.Equal(32, result.Weights[2]);
        Assert.Equal(95, result.Weights[3]);
    }

    [Fact]
    public void Int8_PerRowScale_MultipleRows_IndependentScales()
    {
        // Row 0: [10, -5] → maxAbs=10, scale=10/127
        // Row 1: [0.1, -0.05] → maxAbs=0.1, scale=0.1/127
        float[] data = { 10f, -5f, 0.1f, -0.05f };
        var result = Int8WeightOnlyQuantization.QuantizePerRow(data.AsSpan(), rows: 2, cols: 2);

        float expectedScale0 = 10f / 127f;
        float expectedScale1 = 0.1f / 127f;
        Assert.Equal(expectedScale0, result.Scales[0], 1e-6);
        Assert.Equal(expectedScale1, result.Scales[1], 1e-6);

        // Row 0: 10 * (127/10) = 127, -5 * (127/10) = -63.5 → round = -64
        Assert.Equal(127, result.Weights[0]);
        Assert.Equal(-64, result.Weights[1]);

        // Row 1: 0.1 * (127/0.1) = 127, -0.05 * (127/0.1) = -63.5 → round = -64
        Assert.Equal(127, result.Weights[2]);
        Assert.Equal(-64, result.Weights[3]);
    }

    [Fact]
    public void Int8_RoundTrip_SmallError()
    {
        // Quantize and dequantize should introduce only small quantization error
        // Error bound: maxAbs / 127 per element (half a quantization step)
        float[] original = { 0.3f, -0.7f, 1.5f, -2.3f, 0.0f, 4.0f };
        var qResult = Int8WeightOnlyQuantization.QuantizePerRow(original.AsSpan(), rows: 2, cols: 3);

        // Dequantize: value = weight * scale
        for (int r = 0; r < 2; r++)
        {
            float scale = qResult.Scales[r];
            float maxAbs = 0f;
            for (int c = 0; c < 3; c++)
            {
                float v = MathF.Abs(original[r * 3 + c]);
                if (v > maxAbs) maxAbs = v;
            }

            float maxError = maxAbs / 127f; // Quantization step size

            for (int c = 0; c < 3; c++)
            {
                float dequantized = qResult.Weights[r * 3 + c] * scale;
                float error = MathF.Abs(dequantized - original[r * 3 + c]);
                Assert.True(error <= maxError + 1e-5f,
                    $"Round-trip error {error} exceeds max {maxError} for [{r},{c}]");
            }
        }
    }

    [Fact]
    public void Int8_ZeroRow_ScaleIsOne()
    {
        // All zeros: maxAbs=0, scale=1 (fallback)
        float[] data = { 0f, 0f, 0f };
        var result = Int8WeightOnlyQuantization.QuantizePerRow(data.AsSpan(), rows: 1, cols: 3);

        Assert.Equal(1f, result.Scales[0]);
        Assert.Equal(0, result.Weights[0]);
        Assert.Equal(0, result.Weights[1]);
        Assert.Equal(0, result.Weights[2]);
    }

    [Fact]
    public void Int8_Clamping_LargeValues()
    {
        // Values that would exceed [-127, 127] after scaling should be clamped
        // Row: [100, -100, 50, -50]: maxAbs=100, scale=100/127
        // 100 * (127/100) = 127 (exact)
        // 50 * (127/100) = 63.5 → round = 64
        float[] data = { 100f, -100f, 50f, -50f };
        var result = Int8WeightOnlyQuantization.QuantizePerRow(data.AsSpan(), rows: 1, cols: 4);

        Assert.Equal(127, result.Weights[0]);
        Assert.Equal(-127, result.Weights[1]);
        Assert.Equal(64, result.Weights[2]);
        Assert.Equal(-64, result.Weights[3]);
    }

    [Fact]
    public void Int8_TensorOverload_MatchesSpanOverload()
    {
        float[] data = { 1.0f, -0.5f, 0.3f, -0.8f, 0.2f, 0.9f };
        var tensor = new Tensor<float>(new[] { 2, 3 });
        for (int i = 0; i < 6; i++)
        {
            tensor[i / 3, i % 3] = data[i];
        }

        var tensorResult = Int8WeightOnlyQuantization.QuantizePerRow(tensor);
        var spanResult = Int8WeightOnlyQuantization.QuantizePerRow(data.AsSpan(), rows: 2, cols: 3);

        // Both should produce identical results
        Assert.Equal(tensorResult.Rows, spanResult.Rows);
        Assert.Equal(tensorResult.Cols, spanResult.Cols);
        for (int i = 0; i < tensorResult.Weights.Length; i++)
        {
            Assert.Equal(tensorResult.Weights[i], spanResult.Weights[i]);
        }
        for (int i = 0; i < tensorResult.Scales.Length; i++)
        {
            Assert.Equal(tensorResult.Scales[i], spanResult.Scales[i]);
        }
    }

    [Fact]
    public void Int8_InvalidDimensions_Throws()
    {
        float[] data = { 1f, 2f, 3f };

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            Int8WeightOnlyQuantization.QuantizePerRow(data.AsSpan(), rows: 0, cols: 3));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            Int8WeightOnlyQuantization.QuantizePerRow(data.AsSpan(), rows: 1, cols: 0));
    }

    [Fact]
    public void Int8_SpanTooSmall_Throws()
    {
        float[] data = { 1f, 2f };

        Assert.Throws<ArgumentException>(() =>
            Int8WeightOnlyQuantization.QuantizePerRow(data.AsSpan(), rows: 2, cols: 3));
    }

    [Fact]
    public void Int8_TensorNot2D_Throws()
    {
        var tensor1d = new Tensor<float>(new[] { 4 });

        Assert.Throws<ArgumentException>(() => Int8WeightOnlyQuantization.QuantizePerRow(tensor1d));
    }

    [Fact]
    public void Int8_SymmetricValues_SymmetricQuantization()
    {
        // +X and -X should quantize to +q and -q (symmetric)
        float[] data = { 2.5f, -2.5f, 1.25f, -1.25f };
        var result = Int8WeightOnlyQuantization.QuantizePerRow(data.AsSpan(), rows: 1, cols: 4);

        Assert.Equal(-result.Weights[0], result.Weights[1]);
        Assert.Equal(-result.Weights[2], result.Weights[3]);
    }

    #endregion

    #region FP8 E4M3 Encoding/Decoding Golden References

    [Fact]
    public void FP8_E4M3_Zero_EncodesDecode()
    {
        byte encoded = FP8Quantizer<float, float[], float[]>.E4M3ToByte(0.0);
        Assert.Equal(0, encoded);

        double decoded = FP8Quantizer<float, float[], float[]>.ByteToE4M3(0);
        Assert.Equal(0.0, decoded);
    }

    [Fact]
    public void FP8_E4M3_One_GoldenReference()
    {
        // 1.0 = sign:0, exp:0111 (7=bias), mantissa:000
        // byte = 0|0111|000 = 0x38
        byte encoded = FP8Quantizer<float, float[], float[]>.E4M3ToByte(1.0);
        Assert.Equal(0x38, encoded);

        double decoded = FP8Quantizer<float, float[], float[]>.ByteToE4M3(0x38);
        Assert.Equal(1.0, decoded, Tolerance);
    }

    [Fact]
    public void FP8_E4M3_NegativeOne_GoldenReference()
    {
        // -1.0 = sign:1, exp:0111, mantissa:000
        // byte = 1|0111|000 = 0xB8
        byte encoded = FP8Quantizer<float, float[], float[]>.E4M3ToByte(-1.0);
        Assert.Equal(0xB8, encoded);

        double decoded = FP8Quantizer<float, float[], float[]>.ByteToE4M3(0xB8);
        Assert.Equal(-1.0, decoded, Tolerance);
    }

    [Fact]
    public void FP8_E4M3_MaxValue_Is448()
    {
        // E4M3 max = 448 (per NVIDIA spec)
        // exp=15 (max normal), mantissa=110 (6/8 = 0.75) since 111=7 is NaN
        // value = (1 + 6/8) * 2^(15-7) = 1.75 * 256 = 448
        double decoded = FP8Quantizer<float, float[], float[]>.ByteToE4M3(0x7E);
        Assert.Equal(448.0, decoded, Tolerance);
    }

    [Fact]
    public void FP8_E4M3_NaN_Encoding()
    {
        // NaN = exp=15, mantissa=7 → byte = 0|1111|111 = 0x7F
        byte encoded = FP8Quantizer<float, float[], float[]>.E4M3ToByte(double.NaN);
        Assert.Equal(0x7F, encoded);

        double decoded = FP8Quantizer<float, float[], float[]>.ByteToE4M3(0x7F);
        Assert.True(double.IsNaN(decoded));
    }

    [Fact]
    public void FP8_E4M3_RoundTrip_CommonValues()
    {
        // Test round-trip for values representable exactly in E4M3
        double[] exactValues = { 0.5, 1.0, 1.5, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 448.0 };

        foreach (var original in exactValues)
        {
            byte encoded = FP8Quantizer<float, float[], float[]>.E4M3ToByte(original);
            double decoded = FP8Quantizer<float, float[], float[]>.ByteToE4M3(encoded);

            Assert.Equal(original, decoded, Tolerance);
        }
    }

    [Fact]
    public void FP8_E4M3_RoundTrip_NegativeValues()
    {
        double[] negValues = { -0.5, -1.0, -2.0, -128.0, -448.0 };

        foreach (var original in negValues)
        {
            byte encoded = FP8Quantizer<float, float[], float[]>.E4M3ToByte(original);
            double decoded = FP8Quantizer<float, float[], float[]>.ByteToE4M3(encoded);

            Assert.Equal(original, decoded, Tolerance);
        }
    }

    [Fact]
    public void FP8_E4M3_Overflow_ClampedToMax()
    {
        // Values > 448 should be clamped to 448
        byte encoded = FP8Quantizer<float, float[], float[]>.E4M3ToByte(1000.0);
        double decoded = FP8Quantizer<float, float[], float[]>.ByteToE4M3(encoded);
        Assert.Equal(448.0, decoded, Tolerance);

        encoded = FP8Quantizer<float, float[], float[]>.E4M3ToByte(-1000.0);
        decoded = FP8Quantizer<float, float[], float[]>.ByteToE4M3(encoded);
        Assert.Equal(-448.0, decoded, Tolerance);
    }

    [Fact]
    public void FP8_E4M3_Subnormal_SmallValues()
    {
        // Subnormal: exp=0, mantissa != 0
        // value = mantissa/8 * 2^(-6)
        // mantissa=1: 1/8 * 2^-6 = 0.001953125
        double minSubnormal = 0.001953125;
        byte encoded = FP8Quantizer<float, float[], float[]>.E4M3ToByte(minSubnormal);
        double decoded = FP8Quantizer<float, float[], float[]>.ByteToE4M3(encoded);

        Assert.Equal(minSubnormal, decoded, 1e-8);
    }

    [Fact]
    public void FP8_E4M3_Underflow_BecomesZero()
    {
        // Very small values below min subnormal should underflow to zero
        byte encoded = FP8Quantizer<float, float[], float[]>.E4M3ToByte(1e-10);
        Assert.Equal(0, encoded);
    }

    #endregion

    #region FP8 E5M2 Encoding/Decoding Golden References

    [Fact]
    public void FP8_E5M2_Zero_EncodesDecodes()
    {
        byte encoded = FP8Quantizer<float, float[], float[]>.E5M2ToByte(0.0);
        Assert.Equal(0, encoded);

        double decoded = FP8Quantizer<float, float[], float[]>.ByteToE5M2(0);
        Assert.Equal(0.0, decoded);
    }

    [Fact]
    public void FP8_E5M2_One_GoldenReference()
    {
        // 1.0 = sign:0, exp:01111 (15=bias), mantissa:00
        // byte = 0|01111|00 = 0x3C
        byte encoded = FP8Quantizer<float, float[], float[]>.E5M2ToByte(1.0);
        Assert.Equal(0x3C, encoded);

        double decoded = FP8Quantizer<float, float[], float[]>.ByteToE5M2(0x3C);
        Assert.Equal(1.0, decoded, Tolerance);
    }

    [Fact]
    public void FP8_E5M2_NaN_Encoding()
    {
        byte encoded = FP8Quantizer<float, float[], float[]>.E5M2ToByte(double.NaN);
        // NaN pattern
        double decoded = FP8Quantizer<float, float[], float[]>.ByteToE5M2(encoded);
        Assert.True(double.IsNaN(decoded));
    }

    [Fact]
    public void FP8_E5M2_Infinity_Encoding()
    {
        // +Inf = exp=31, mantissa=0 → byte = 0|11111|00 = 0x7C
        byte posInf = FP8Quantizer<float, float[], float[]>.E5M2ToByte(double.PositiveInfinity);
        Assert.Equal(0x7C, posInf);

        double decodedPos = FP8Quantizer<float, float[], float[]>.ByteToE5M2(0x7C);
        Assert.True(double.IsPositiveInfinity(decodedPos));

        // -Inf = 1|11111|00 = 0xFC
        byte negInf = FP8Quantizer<float, float[], float[]>.E5M2ToByte(double.NegativeInfinity);
        Assert.Equal(0xFC, negInf);

        double decodedNeg = FP8Quantizer<float, float[], float[]>.ByteToE5M2(0xFC);
        Assert.True(double.IsNegativeInfinity(decodedNeg));
    }

    [Fact]
    public void FP8_E5M2_MaxValue_Is57344()
    {
        // E5M2 max = 57344 per spec
        // exp=30 (max normal), mantissa=11 (3/4 = 0.75)
        // value = (1 + 3/4) * 2^(30-15) = 1.75 * 32768 = 57344
        double decoded = FP8Quantizer<float, float[], float[]>.ByteToE5M2(0x7B);
        Assert.Equal(57344.0, decoded, Tolerance);
    }

    [Fact]
    public void FP8_E5M2_RoundTrip_CommonValues()
    {
        // E5M2 has 2 mantissa bits → representable: 1.0, 1.25, 1.5, 1.75 * 2^exp
        double[] exactValues = { 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0, 16.0, 256.0, 1024.0 };

        foreach (var original in exactValues)
        {
            byte encoded = FP8Quantizer<float, float[], float[]>.E5M2ToByte(original);
            double decoded = FP8Quantizer<float, float[], float[]>.ByteToE5M2(encoded);
            Assert.Equal(original, decoded, Tolerance);
        }
    }

    #endregion

    #region FP8 Weight-Only Quantization Round-Trip Tests

    [Fact]
    public void FP8_WeightOnly_PerRowScale_GoldenReference()
    {
        // Row [10.0, -5.0, 2.5]: maxAbs=10, scale=10/448
        // Scaled values: 10*(448/10)=448, -5*(448/10)=-224, 2.5*(448/10)=112
        float[] data = { 10.0f, -5.0f, 2.5f };
        var result = FP8WeightOnlyQuantization.QuantizePerRow(data.AsSpan(), rows: 1, cols: 3);

        float expectedScale = 10.0f / 448.0f;
        Assert.Equal(expectedScale, result.Scales[0], 1e-5);
        Assert.Equal(1, result.Rows);
        Assert.Equal(3, result.Cols);
    }

    [Fact]
    public void FP8_WeightOnly_RoundTrip_SmallError()
    {
        // Quantize and dequantize should preserve values within FP8 precision
        float[] original = { 1.0f, -0.5f, 0.25f, -2.0f, 0.0f, 3.5f };
        var qResult = FP8WeightOnlyQuantization.QuantizePerRow(original.AsSpan(), rows: 2, cols: 3);

        for (int r = 0; r < 2; r++)
        {
            float scale = qResult.Scales[r];
            for (int c = 0; c < 3; c++)
            {
                int idx = r * 3 + c;
                float dequantized = FP8WeightOnlyQuantization.Dequantize(qResult.Weights[idx], scale);
                float error = MathF.Abs(dequantized - original[idx]);

                // FP8 E4M3 has 3 mantissa bits → ~12.5% relative error max
                // But per-row scaling reduces this for within-row values
                float relTolerance = MathF.Abs(original[idx]) * 0.15f + 0.01f;
                Assert.True(error <= relTolerance,
                    $"FP8 round-trip error {error} > tolerance {relTolerance} for [{r},{c}] (orig={original[idx]}, deq={dequantized})");
            }
        }
    }

    [Fact]
    public void FP8_WeightOnly_ZeroRow_ScaleIsOne()
    {
        float[] data = { 0f, 0f, 0f };
        var result = FP8WeightOnlyQuantization.QuantizePerRow(data.AsSpan(), rows: 1, cols: 3);

        Assert.Equal(1f, result.Scales[0]);
    }

    [Fact]
    public void FP8_WeightOnly_InvalidDimensions_Throws()
    {
        float[] data = { 1f, 2f, 3f };

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            FP8WeightOnlyQuantization.QuantizePerRow(data.AsSpan(), rows: 0, cols: 3));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            FP8WeightOnlyQuantization.QuantizePerRow(data.AsSpan(), rows: 1, cols: 0));
    }

    [Fact]
    public void FP8_WeightOnly_SpanTooSmall_Throws()
    {
        float[] data = { 1f };

        Assert.Throws<ArgumentException>(() =>
            FP8WeightOnlyQuantization.QuantizePerRow(data.AsSpan(), rows: 2, cols: 3));
    }

    [Fact]
    public void FP8_Dequantize_GoldenReference()
    {
        // Manually encode 1.0 as E4M3, then dequantize with scale=2.0
        // E4M3ToByte(1.0) = 0x38, ByteToE4M3(0x38) = 1.0
        // Dequantize: 1.0 * 2.0 = 2.0
        byte fp8Byte = FP8Quantizer<float, float[], float[]>.E4M3ToByte(1.0);
        float dequantized = FP8WeightOnlyQuantization.Dequantize(fp8Byte, 2.0f);
        Assert.Equal(2.0f, dequantized, 1e-4);
    }

    #endregion

    #region INT8 vs FP8 Comparison Tests

    [Fact]
    public void Int8VsFP8_SimilarScaleForUniformData()
    {
        // For uniform data, both should produce similar scales
        float[] data = { 1.0f, -0.5f, 0.3f, -0.8f };

        var int8Result = Int8WeightOnlyQuantization.QuantizePerRow(data.AsSpan(), rows: 1, cols: 4);
        var fp8Result = FP8WeightOnlyQuantization.QuantizePerRow(data.AsSpan(), rows: 1, cols: 4);

        // INT8 scale = maxAbs/127, FP8 scale = maxAbs/448
        float expectedInt8Scale = 1.0f / 127f;
        float expectedFP8Scale = 1.0f / 448f;

        Assert.Equal(expectedInt8Scale, int8Result.Scales[0], 1e-6);
        Assert.Equal(expectedFP8Scale, fp8Result.Scales[0], 1e-6);

        // FP8 scale should be smaller (tighter packing)
        Assert.True(fp8Result.Scales[0] < int8Result.Scales[0]);
    }

    [Fact]
    public void Int8_LargeMatrix_AllValuesInRange()
    {
        // Stress test: 10x10 matrix with varying magnitudes
        int rows = 10, cols = 10;
        var data = new float[rows * cols];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = (float)Math.Sin(i * 0.5) * (i % 7 + 1);
        }

        var result = Int8WeightOnlyQuantization.QuantizePerRow(data.AsSpan(), rows, cols);

        // All quantized values must be in [-127, 127]
        for (int i = 0; i < result.Weights.Length; i++)
        {
            Assert.InRange(result.Weights[i], (sbyte)-127, (sbyte)127);
        }

        // All scales must be positive
        for (int i = 0; i < result.Scales.Length; i++)
        {
            Assert.True(result.Scales[i] > 0f);
        }
    }

    #endregion
}
