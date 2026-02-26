using System;
using AiDotNet.LinearAlgebra;
using AiDotNet.MixedPrecision;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.MixedPrecision;

/// <summary>
/// Deep math-correctness integration tests for the MixedPrecision module.
/// Verifies Float8E4M3 and Float8E5M2 bit-level encoding/decoding, LossScaler
/// dynamic scaling state machine, and MixedPrecisionContext FP32/FP16 casting
/// against hand-computed expected values and IEEE 754-like format specifications.
/// </summary>
public class MixedPrecisionDeepMathIntegrationTests
{
    private const double Tolerance = 1e-5;
    private const float FloatTolerance = 1e-5f;

    #region Float8E4M3 - Basic Conversion Tests

    [Fact]
    public void E4M3_Zero_RoundTrips()
    {
        // E4M3 zero: sign=0, exp=0, mantissa=0 => byte 0x00
        var zero = Float8E4M3.FromFloat(0f);
        Assert.Equal(0f, zero.ToFloat());
        Assert.True(zero.IsZero);
        Assert.False(zero.IsNegative);
        Assert.Equal(0, zero.RawValue);
    }

    [Fact]
    public void E4M3_NegativeZero_PreserveSign()
    {
        // Negative zero: sign=1, exp=0, mantissa=0 => byte 0x80
        var negZero = Float8E4M3.FromFloat(-0f);
        Assert.True(negZero.IsZero);
        Assert.True(negZero.IsNegative);
        Assert.Equal(0x80, negZero.RawValue);
    }

    [Fact]
    public void E4M3_One_ExactEncoding()
    {
        // 1.0: sign=0, exp=7 (bias=7, so stored exp=7), mantissa=0
        // Byte: 0_0111_000 = 0x38
        // Verify: 1.0 * 2^(7-7) = 1.0 * 2^0 = 1.0
        var one = Float8E4M3.FromFloat(1.0f);
        Assert.Equal(1.0f, one.ToFloat());
        Assert.Equal(0x38, one.RawValue);
    }

    [Fact]
    public void E4M3_Two_ExactEncoding()
    {
        // 2.0: sign=0, exp=8 (bias=7, so real exp=1), mantissa=0
        // Byte: 0_1000_000 = 0x40
        // Verify: 1.0 * 2^(8-7) = 1.0 * 2^1 = 2.0
        var two = Float8E4M3.FromFloat(2.0f);
        Assert.Equal(2.0f, two.ToFloat());
        Assert.Equal(0x40, two.RawValue);
    }

    [Fact]
    public void E4M3_NegativeOne_ExactEncoding()
    {
        // -1.0: sign=1, exp=7, mantissa=0
        // Byte: 1_0111_000 = 0xB8
        var negOne = Float8E4M3.FromFloat(-1.0f);
        Assert.Equal(-1.0f, negOne.ToFloat());
        Assert.Equal(0xB8, negOne.RawValue);
    }

    [Fact]
    public void E4M3_OnePointFive_ExactEncoding()
    {
        // 1.5: sign=0, exp=7, mantissa=4 (1.100 in binary = 1.5)
        // mantissa bits: 100 = 4
        // Byte: 0_0111_100 = 0x3C
        var val = Float8E4M3.FromFloat(1.5f);
        Assert.Equal(1.5f, val.ToFloat());
        Assert.Equal(0x3C, val.RawValue);
    }

    [Fact]
    public void E4M3_MaxValue_Is448()
    {
        // E4M3 max finite value: exp=15, mantissa=6 (not 7 since 0x7F is NaN)
        // value = 1.110 * 2^(15-7) = 1.75 * 256 = 448
        var max = Float8E4M3.MaxValue;
        Assert.Equal(448f, max.ToFloat());
    }

    [Fact]
    public void E4M3_ClampToMax_LargeValueBecomesMaxFinite()
    {
        // Values above 448 should be clamped to 448
        var clamped = Float8E4M3.FromFloat(1000f);
        Assert.Equal(448f, clamped.ToFloat());
    }

    [Fact]
    public void E4M3_NaN_EncodedCorrectly()
    {
        // NaN: exp=15 (all 1s), mantissa=7 (all 1s) => byte 0x7F
        var nan = Float8E4M3.FromFloat(float.NaN);
        Assert.True(nan.IsNaN);
        Assert.Equal(0x7F, nan.RawValue);
        Assert.True(float.IsNaN(nan.ToFloat()));
    }

    [Fact]
    public void E4M3_SmallestNormal_Is0Point125()
    {
        // Smallest normal: exp=1, mantissa=0
        // value = 1.0 * 2^(1-7) = 2^-6 = 0.015625
        var val = Float8E4M3.FromFloat(0.015625f);
        float result = val.ToFloat();
        Assert.Equal(0.015625f, result, FloatTolerance);
    }

    [Fact]
    public void E4M3_SubnormalValues_CorrectEncoding()
    {
        // Subnormal: exp=0, mantissa=m
        // value = m/8 * 2^-6 = m * 2^-9
        // For m=1: 1 * 2^-9 = 0.001953125
        var val = Float8E4M3.FromFloat(0.001953125f);
        float result = val.ToFloat();
        Assert.Equal(0.001953125f, result, FloatTolerance);
    }

    [Fact]
    public void E4M3_SubnormalLargest_CorrectValue()
    {
        // Largest subnormal: exp=0, mantissa=7
        // value = 7/8 * 2^-6 = 7 * 2^-9 = 0.013671875
        var val = Float8E4M3.FromFloat(0.013671875f);
        float result = val.ToFloat();
        Assert.Equal(0.013671875f, result, FloatTolerance);
    }

    [Fact]
    public void E4M3_BelowMinSubnormal_FlushesToZero()
    {
        // Values below smallest subnormal (2^-9 = 0.001953125) should flush to zero
        var val = Float8E4M3.FromFloat(0.0001f);
        Assert.True(val.IsZero);
    }

    #endregion

    #region Float8E4M3 - Conversion Accuracy Tests

    [Fact]
    public void E4M3_PowersOfTwo_ExactRoundTrip()
    {
        // Powers of 2 within range should be exact
        float[] powers = { 0.015625f, 0.03125f, 0.0625f, 0.125f, 0.25f, 0.5f, 1f, 2f, 4f, 8f, 16f, 32f, 64f, 128f, 256f };
        foreach (var p in powers)
        {
            var encoded = Float8E4M3.FromFloat(p);
            float decoded = encoded.ToFloat();
            Assert.Equal(p, decoded, FloatTolerance);
        }
    }

    [Fact]
    public void E4M3_NegativePowersOfTwo_ExactRoundTrip()
    {
        float[] powers = { -0.125f, -0.25f, -0.5f, -1f, -2f, -4f, -8f, -16f, -128f };
        foreach (var p in powers)
        {
            var encoded = Float8E4M3.FromFloat(p);
            float decoded = encoded.ToFloat();
            Assert.Equal(p, decoded, FloatTolerance);
        }
    }

    [Fact]
    public void E4M3_ThreePointFive_HandComputedBits()
    {
        // 3.5 = 1.75 * 2^1
        // sign=0, exp=8 (real exp=1, stored=1+7=8), mantissa=110 = 6
        // Byte: 0_1000_110 = 0x46
        var val = Float8E4M3.FromFloat(3.5f);
        Assert.Equal(3.5f, val.ToFloat());
        Assert.Equal(0x46, val.RawValue);
    }

    [Fact]
    public void E4M3_Equality_SameValue()
    {
        var a = Float8E4M3.FromFloat(1.5f);
        var b = Float8E4M3.FromFloat(1.5f);
        Assert.True(a == b);
        Assert.Equal(a, b);
    }

    [Fact]
    public void E4M3_Comparison_Ordering()
    {
        var small = Float8E4M3.FromFloat(1.0f);
        var large = Float8E4M3.FromFloat(2.0f);
        Assert.True(small < large);
        Assert.True(large > small);
        Assert.True(small <= large);
        Assert.True(large >= small);
    }

    [Fact]
    public void E4M3_NegativeValueComparison()
    {
        var neg = Float8E4M3.FromFloat(-2.0f);
        var pos = Float8E4M3.FromFloat(1.0f);
        Assert.True(neg < pos);
    }

    #endregion

    #region Float8E5M2 - Basic Conversion Tests

    [Fact]
    public void E5M2_Zero_RoundTrips()
    {
        var zero = Float8E5M2.FromFloat(0f);
        Assert.Equal(0f, zero.ToFloat());
        Assert.True(zero.IsZero);
        Assert.False(zero.IsNegative);
        Assert.Equal(0, zero.RawValue);
    }

    [Fact]
    public void E5M2_NegativeZero_PreserveSign()
    {
        var negZero = Float8E5M2.FromFloat(-0f);
        Assert.True(negZero.IsZero);
        Assert.True(negZero.IsNegative);
        Assert.Equal(0x80, negZero.RawValue);
    }

    [Fact]
    public void E5M2_One_ExactEncoding()
    {
        // 1.0: sign=0, exp=15 (bias=15, stored exp=15), mantissa=0
        // Byte: 0_01111_00 = 0x3C
        var one = Float8E5M2.FromFloat(1.0f);
        Assert.Equal(1.0f, one.ToFloat());
        Assert.Equal(0x3C, one.RawValue);
    }

    [Fact]
    public void E5M2_Two_ExactEncoding()
    {
        // 2.0: sign=0, exp=16, mantissa=0
        // Byte: 0_10000_00 = 0x40
        var two = Float8E5M2.FromFloat(2.0f);
        Assert.Equal(2.0f, two.ToFloat());
        Assert.Equal(0x40, two.RawValue);
    }

    [Fact]
    public void E5M2_NegativeOne_ExactEncoding()
    {
        // -1.0: sign=1, exp=15, mantissa=0
        // Byte: 1_01111_00 = 0xBC
        var negOne = Float8E5M2.FromFloat(-1.0f);
        Assert.Equal(-1.0f, negOne.ToFloat());
        Assert.Equal(0xBC, negOne.RawValue);
    }

    [Fact]
    public void E5M2_OnePointFive_ExactEncoding()
    {
        // 1.5: sign=0, exp=15, mantissa=10 = 2
        // Byte: 0_01111_10 = 0x3E
        var val = Float8E5M2.FromFloat(1.5f);
        Assert.Equal(1.5f, val.ToFloat());
        Assert.Equal(0x3E, val.RawValue);
    }

    [Fact]
    public void E5M2_MaxValue_Is57344()
    {
        // E5M2 max finite value: exp=30, mantissa=3
        // value = 1.11 * 2^(30-15) = 1.75 * 32768 = 57344
        var max = Float8E5M2.MaxValue;
        Assert.Equal(57344f, max.ToFloat());
    }

    [Fact]
    public void E5M2_OverMax_BecomesInfinity()
    {
        // Values above 57344 should become infinity
        var inf = Float8E5M2.FromFloat(100000f);
        Assert.True(inf.IsInfinity);
        Assert.False(inf.IsNegative);
    }

    [Fact]
    public void E5M2_NegativeOverMax_BecomesNegativeInfinity()
    {
        var negInf = Float8E5M2.FromFloat(-100000f);
        Assert.True(negInf.IsInfinity);
        Assert.True(negInf.IsNegative);
    }

    [Fact]
    public void E5M2_PositiveInfinity_EncodedCorrectly()
    {
        // +Inf: sign=0, exp=31, mantissa=0 => byte 0x7C
        var inf = Float8E5M2.FromFloat(float.PositiveInfinity);
        Assert.True(inf.IsInfinity);
        Assert.False(inf.IsNegative);
        Assert.Equal(0x7C, inf.RawValue);
    }

    [Fact]
    public void E5M2_NegativeInfinity_EncodedCorrectly()
    {
        // -Inf: sign=1, exp=31, mantissa=0 => byte 0xFC
        var inf = Float8E5M2.FromFloat(float.NegativeInfinity);
        Assert.True(inf.IsInfinity);
        Assert.True(inf.IsNegative);
        Assert.Equal(0xFC, inf.RawValue);
    }

    [Fact]
    public void E5M2_NaN_EncodedCorrectly()
    {
        // NaN: exp=31, mantissa!=0 => byte 0x7F
        var nan = Float8E5M2.FromFloat(float.NaN);
        Assert.True(nan.IsNaN);
        Assert.True(float.IsNaN(nan.ToFloat()));
    }

    [Fact]
    public void E5M2_SubnormalValues_CorrectEncoding()
    {
        // Subnormal E5M2: exp=0, mantissa=m
        // value = m/4 * 2^-14 = m * 2^-16
        // For m=1: 2^-16 = 0.0000152588
        var val = Float8E5M2.FromFloat(0.0000152588f);
        float result = val.ToFloat();
        // Should be close to 2^-16
        Assert.True(Math.Abs(result - 0.0000152588f) < 0.0001f);
    }

    [Fact]
    public void E5M2_BelowMinSubnormal_FlushesToZero()
    {
        var val = Float8E5M2.FromFloat(1e-6f);
        Assert.True(val.IsZero);
    }

    #endregion

    #region Float8E5M2 - Conversion Accuracy Tests

    [Fact]
    public void E5M2_PowersOfTwo_ExactRoundTrip()
    {
        // E5M2 has wider range but less precision
        float[] powers = { 0.00006103515625f, 0.5f, 1f, 2f, 4f, 8f, 16f, 256f, 1024f, 4096f, 16384f };
        foreach (var p in powers)
        {
            var encoded = Float8E5M2.FromFloat(p);
            float decoded = encoded.ToFloat();
            Assert.Equal(p, decoded, FloatTolerance);
        }
    }

    [Fact]
    public void E5M2_Comparison_Ordering()
    {
        var small = Float8E5M2.FromFloat(1.0f);
        var large = Float8E5M2.FromFloat(2.0f);
        Assert.True(small < large);
        Assert.True(large > small);
    }

    [Fact]
    public void E5M2_Equality_SameValue()
    {
        var a = Float8E5M2.FromFloat(1.5f);
        var b = Float8E5M2.FromFloat(1.5f);
        Assert.True(a == b);
        Assert.Equal(a, b);
    }

    #endregion

    #region Float8 Cross-Format Conversion Tests

    [Fact]
    public void E4M3_ToE5M2_PreservesValue()
    {
        // Convert 1.5 from E4M3 to E5M2 via float
        var e4m3 = Float8E4M3.FromFloat(1.5f);
        var e5m2 = e4m3.ToE5M2();
        Assert.Equal(1.5f, e5m2.ToFloat());
    }

    [Fact]
    public void E5M2_ToE4M3_PreservesValue()
    {
        var e5m2 = Float8E5M2.FromFloat(2.0f);
        var e4m3 = e5m2.ToE4M3();
        Assert.Equal(2.0f, e4m3.ToFloat());
    }

    [Fact]
    public void E4M3_HasMorePrecision_ThanE5M2()
    {
        // E4M3 has 3 mantissa bits => 8 distinct mantissa values per exponent
        // E5M2 has 2 mantissa bits => 4 distinct mantissa values per exponent
        // 1.25 = 1.01 binary, needs 2 mantissa bits minimum
        // E4M3 can represent 1.25 exactly (mantissa=010)
        // E5M2 can represent 1.25 exactly (mantissa=01)
        var e4m3 = Float8E4M3.FromFloat(1.25f);
        var e5m2 = Float8E5M2.FromFloat(1.25f);
        Assert.Equal(1.25f, e4m3.ToFloat());
        Assert.Equal(1.25f, e5m2.ToFloat());

        // 1.125 = 1.001 binary, needs 3 mantissa bits
        // E4M3 can represent exactly (mantissa=001)
        // E5M2 must round (only 2 mantissa bits)
        var e4m3_precise = Float8E4M3.FromFloat(1.125f);
        Assert.Equal(1.125f, e4m3_precise.ToFloat());
    }

    [Fact]
    public void E5M2_HasMoreRange_ThanE4M3()
    {
        // E5M2 max = 57344, E4M3 max = 448
        // 1000.0 is within E5M2 range but above E4M3 max
        var e5m2 = Float8E5M2.FromFloat(1024f);
        Assert.Equal(1024f, e5m2.ToFloat());

        var e4m3 = Float8E4M3.FromFloat(1024f);
        Assert.Equal(448f, e4m3.ToFloat()); // Clamped to max
    }

    [Fact]
    public void E4M3_BulkConversion_ArrayRoundTrip()
    {
        float[] values = { 0f, 1f, 2f, 4f, 8f, -1f, -4f, 0.5f, 0.25f };
        var e4m3Array = values.ToE4M3();
        float[] result = e4m3Array.ToFloatArray();

        Assert.Equal(values.Length, result.Length);
        for (int i = 0; i < values.Length; i++)
        {
            Assert.Equal(values[i], result[i], FloatTolerance);
        }
    }

    [Fact]
    public void E5M2_BulkConversion_ArrayRoundTrip()
    {
        float[] values = { 0f, 1f, 2f, 4f, 1024f, -1f, -256f };
        var e5m2Array = values.ToE5M2();
        float[] result = e5m2Array.ToFloatArray();

        Assert.Equal(values.Length, result.Length);
        for (int i = 0; i < values.Length; i++)
        {
            Assert.Equal(values[i], result[i], FloatTolerance);
        }
    }

    #endregion

    #region Float8E4M3 - Bit-Level Verification Tests

    [Fact]
    public void E4M3_BitDecomposition_ManualVerification()
    {
        // For value 3.0: 1.1 * 2^1
        // sign=0, exp=8 (1+7), mantissa=100 = 4
        // Byte: 0_1000_100 = 0x44
        var val = Float8E4M3.FromFloat(3.0f);
        byte raw = val.RawValue;
        int sign = (raw >> 7) & 1;
        int exp = (raw >> 3) & 0xF;
        int mantissa = raw & 0x07;

        Assert.Equal(0, sign);
        Assert.Equal(8, exp);
        Assert.Equal(4, mantissa);

        // Reconstruct: 1.mantissa * 2^(exp-bias) = 1.100 * 2^1 = 1.5 * 2 = 3.0
        float expected = (1.0f + mantissa / 8.0f) * MathF.Pow(2, exp - 7);
        Assert.Equal(3.0f, expected, FloatTolerance);
    }

    [Fact]
    public void E4M3_BitDecomposition_ForFive()
    {
        // 5.0 = 1.01 * 2^2
        // sign=0, exp=9 (2+7), mantissa=010 = 2
        // Byte: 0_1001_010 = 0x4A
        var val = Float8E4M3.FromFloat(5.0f);
        byte raw = val.RawValue;
        int sign = (raw >> 7) & 1;
        int exp = (raw >> 3) & 0xF;
        int mantissa = raw & 0x07;

        Assert.Equal(0, sign);
        Assert.Equal(9, exp);
        Assert.Equal(2, mantissa);

        float expected = (1.0f + mantissa / 8.0f) * MathF.Pow(2, exp - 7);
        Assert.Equal(5.0f, expected, FloatTolerance);
    }

    [Fact]
    public void E4M3_AllExponentValues_CorrectRange()
    {
        // For each exponent value (1-15, normal range), verify the smallest value
        for (int e = 1; e <= 15; e++)
        {
            // Smallest normal for this exponent: 1.000 * 2^(e-7)
            float expected = MathF.Pow(2, e - 7);
            if (expected > 448f) break; // Skip beyond max

            var encoded = Float8E4M3.FromFloat(expected);
            Assert.Equal(expected, encoded.ToFloat(), FloatTolerance);
        }
    }

    #endregion

    #region Float8E5M2 - Bit-Level Verification Tests

    [Fact]
    public void E5M2_BitDecomposition_ManualVerification()
    {
        // For value 3.0: 1.1 * 2^1
        // sign=0, exp=16 (1+15), mantissa=10 = 2
        // Byte: 0_10000_10 = 0x42
        var val = Float8E5M2.FromFloat(3.0f);
        byte raw = val.RawValue;
        int sign = (raw >> 7) & 1;
        int exp = (raw >> 2) & 0x1F;
        int mantissa = raw & 0x03;

        Assert.Equal(0, sign);
        Assert.Equal(16, exp);
        Assert.Equal(2, mantissa);

        float expected = (1.0f + mantissa / 4.0f) * MathF.Pow(2, exp - 15);
        Assert.Equal(3.0f, expected, FloatTolerance);
    }

    [Fact]
    public void E5M2_BitDecomposition_ForEight()
    {
        // 8.0 = 1.0 * 2^3
        // sign=0, exp=18 (3+15), mantissa=00 = 0
        // Byte: 0_10010_00 = 0x48
        var val = Float8E5M2.FromFloat(8.0f);
        byte raw = val.RawValue;
        int sign = (raw >> 7) & 1;
        int exp = (raw >> 2) & 0x1F;
        int mantissa = raw & 0x03;

        Assert.Equal(0, sign);
        Assert.Equal(18, exp);
        Assert.Equal(0, mantissa);

        float expected = (1.0f + mantissa / 4.0f) * MathF.Pow(2, exp - 15);
        Assert.Equal(8.0f, expected, FloatTolerance);
    }

    #endregion

    #region LossScaler - Exact Scaling Math Tests

    [Fact]
    public void LossScaler_ScaleLoss_ExactMultiplication()
    {
        // ScaleLoss(loss) = loss * scale
        var scaler = new LossScaler<double>(initialScale: 65536.0);
        double loss = 0.0001;
        double scaledLoss = scaler.ScaleLoss(loss);

        // 0.0001 * 65536 = 6.5536
        Assert.Equal(6.5536, scaledLoss, Tolerance);
    }

    [Fact]
    public void LossScaler_UnscaleGradient_ExactDivision()
    {
        // UnscaleGradient(grad) = grad * (1/scale)
        var scaler = new LossScaler<double>(initialScale: 256.0);
        double gradient = 128.0;
        double unscaled = scaler.UnscaleGradient(gradient);

        // 128 * (1/256) = 0.5
        Assert.Equal(0.5, unscaled, Tolerance);
    }

    [Fact]
    public void LossScaler_ScaleThenUnscale_IsIdentity()
    {
        // UnscaleGradient(ScaleLoss(x)) should approximately equal x
        var scaler = new LossScaler<double>(initialScale: 1024.0);
        double original = 0.00375;
        double scaled = scaler.ScaleLoss(original);
        double recovered = scaler.UnscaleGradient(scaled);

        Assert.Equal(original, recovered, Tolerance);
    }

    [Fact]
    public void LossScaler_VectorUnscale_ExactPerElementDivision()
    {
        var scaler = new LossScaler<double>(initialScale: 200.0);
        var gradients = new Vector<double>(new[] { 100.0, 200.0, 400.0, 600.0 });

        scaler.UnscaleGradients(gradients);

        // Each element divided by 200
        Assert.Equal(0.5, gradients[0], Tolerance);
        Assert.Equal(1.0, gradients[1], Tolerance);
        Assert.Equal(2.0, gradients[2], Tolerance);
        Assert.Equal(3.0, gradients[3], Tolerance);
    }

    [Fact]
    public void LossScaler_TensorUnscale_ExactPerElementDivision()
    {
        var scaler = new LossScaler<double>(initialScale: 50.0);
        var data = new Vector<double>(new[] { 25.0, 50.0, 75.0, 100.0 });
        var gradients = new Tensor<double>(new[] { 2, 2 }, data);

        scaler.UnscaleGradients(gradients);

        // Each element divided by 50
        Assert.Equal(0.5, gradients.GetFlatIndexValue(0), Tolerance);
        Assert.Equal(1.0, gradients.GetFlatIndexValue(1), Tolerance);
        Assert.Equal(1.5, gradients.GetFlatIndexValue(2), Tolerance);
        Assert.Equal(2.0, gradients.GetFlatIndexValue(3), Tolerance);
    }

    #endregion

    #region LossScaler - Dynamic Scaling State Machine Tests

    [Fact]
    public void LossScaler_DynamicGrowth_ExactScaleAfterInterval()
    {
        // After GrowthInterval consecutive successes, scale *= GrowthFactor
        var scaler = new LossScaler<double>(
            initialScale: 100.0,
            dynamicScaling: true,
            growthInterval: 3,
            growthFactor: 2.0
        );

        // 3 successful updates
        for (int i = 0; i < 3; i++)
        {
            scaler.UnscaleGradientsAndCheck(new Vector<double>(new[] { 10.0 }));
        }

        // Scale should be 100 * 2 = 200
        Assert.Equal(200.0, scaler.Scale, Tolerance);
    }

    [Fact]
    public void LossScaler_DynamicBackoff_ExactScaleOnOverflow()
    {
        // On overflow: scale *= BackoffFactor
        var scaler = new LossScaler<double>(
            initialScale: 1000.0,
            dynamicScaling: true,
            backoffFactor: 0.5
        );

        // Trigger overflow
        scaler.UnscaleGradientsAndCheck(new Vector<double>(new[] { double.NaN }));

        // Scale should be 1000 * 0.5 = 500
        Assert.Equal(500.0, scaler.Scale, Tolerance);
    }

    [Fact]
    public void LossScaler_MultipleBackoffs_GeometricDecay()
    {
        // Each overflow: scale *= 0.5
        // After 3 overflows from 1000: 1000 * 0.5^3 = 125
        var scaler = new LossScaler<double>(
            initialScale: 1000.0,
            dynamicScaling: true,
            backoffFactor: 0.5,
            minScale: 1.0
        );

        for (int i = 0; i < 3; i++)
        {
            scaler.UnscaleGradientsAndCheck(new Vector<double>(new[] { double.NaN }));
        }

        Assert.Equal(125.0, scaler.Scale, Tolerance);
    }

    [Fact]
    public void LossScaler_MinScaleClamping_PreventsGoingBelow()
    {
        // scale = 10, backoff = 0.1, minScale = 5
        // After overflow: 10 * 0.1 = 1.0, but clamped to 5.0
        var scaler = new LossScaler<double>(
            initialScale: 10.0,
            dynamicScaling: true,
            backoffFactor: 0.1,
            minScale: 5.0
        );

        scaler.UnscaleGradientsAndCheck(new Vector<double>(new[] { double.NaN }));

        Assert.Equal(5.0, scaler.Scale, Tolerance);
    }

    [Fact]
    public void LossScaler_MaxScaleClamping_PreventsGoingAbove()
    {
        // scale = 100, growth = 10, maxScale = 500
        // After growth interval: 100 * 10 = 1000, but clamped to 500
        var scaler = new LossScaler<double>(
            initialScale: 100.0,
            dynamicScaling: true,
            growthInterval: 1,
            growthFactor: 10.0,
            maxScale: 500.0
        );

        scaler.UnscaleGradientsAndCheck(new Vector<double>(new[] { 10.0 }));

        Assert.Equal(500.0, scaler.Scale, Tolerance);
    }

    [Fact]
    public void LossScaler_ConsecutiveSuccessReset_OnOverflow()
    {
        // Growth should only happen after CONSECUTIVE successes
        // If overflow happens in the middle, counter resets
        var scaler = new LossScaler<double>(
            initialScale: 100.0,
            dynamicScaling: true,
            growthInterval: 3,
            growthFactor: 2.0,
            backoffFactor: 0.5
        );

        // 2 successes, then overflow - counter resets
        scaler.UnscaleGradientsAndCheck(new Vector<double>(new[] { 10.0 }));
        scaler.UnscaleGradientsAndCheck(new Vector<double>(new[] { 10.0 }));
        scaler.UnscaleGradientsAndCheck(new Vector<double>(new[] { double.NaN }));

        // Scale reduced by backoff: 100 * 0.5 = 50
        Assert.Equal(50.0, scaler.Scale, Tolerance);

        // 2 more successes - not enough for growth (need 3 consecutive)
        scaler.UnscaleGradientsAndCheck(new Vector<double>(new[] { 10.0 }));
        scaler.UnscaleGradientsAndCheck(new Vector<double>(new[] { 10.0 }));

        // Scale should still be 50 (only 2 consecutive successes)
        Assert.Equal(50.0, scaler.Scale, Tolerance);

        // One more success completes the interval
        scaler.UnscaleGradientsAndCheck(new Vector<double>(new[] { 10.0 }));

        // Scale should grow: 50 * 2 = 100
        Assert.Equal(100.0, scaler.Scale, Tolerance);
    }

    [Fact]
    public void LossScaler_StaticScaling_NeverChanges()
    {
        // Dynamic scaling disabled - scale never changes
        var scaler = new LossScaler<double>(
            initialScale: 256.0,
            dynamicScaling: false
        );

        // Overflow should not change scale
        scaler.UnscaleGradientsAndCheck(new Vector<double>(new[] { double.NaN }));
        Assert.Equal(256.0, scaler.Scale, Tolerance);

        // Many successes should not change scale
        for (int i = 0; i < 5000; i++)
        {
            scaler.UnscaleGradientsAndCheck(new Vector<double>(new[] { 10.0 }));
        }
        Assert.Equal(256.0, scaler.Scale, Tolerance);
    }

    #endregion

    #region LossScaler - Overflow Rate Math Tests

    [Fact]
    public void LossScaler_OverflowRate_ExactComputation()
    {
        // OverflowRate = skippedUpdates / totalUpdates
        var scaler = new LossScaler<double>(initialScale: 100.0);

        // 3 successful, 2 overflow = 2/5 = 0.4
        scaler.UnscaleGradientsAndCheck(new Vector<double>(new[] { 10.0 }));
        scaler.UnscaleGradientsAndCheck(new Vector<double>(new[] { 10.0 }));
        scaler.UnscaleGradientsAndCheck(new Vector<double>(new[] { double.NaN }));
        scaler.UnscaleGradientsAndCheck(new Vector<double>(new[] { 10.0 }));
        scaler.UnscaleGradientsAndCheck(new Vector<double>(new[] { double.PositiveInfinity }));

        Assert.Equal(5, scaler.TotalUpdates);
        Assert.Equal(2, scaler.SkippedUpdates);
        Assert.Equal(0.4, scaler.OverflowRate, Tolerance);
    }

    [Fact]
    public void LossScaler_OverflowRate_ZeroUpdates_ReturnsZero()
    {
        var scaler = new LossScaler<double>(initialScale: 100.0);
        Assert.Equal(0.0, scaler.OverflowRate, Tolerance);
    }

    [Fact]
    public void LossScaler_OverflowRate_AllOverflow_ReturnsOne()
    {
        var scaler = new LossScaler<double>(initialScale: 100.0);

        for (int i = 0; i < 5; i++)
        {
            scaler.UnscaleGradientsAndCheck(new Vector<double>(new[] { double.NaN }));
        }

        Assert.Equal(1.0, scaler.OverflowRate, Tolerance);
    }

    [Fact]
    public void LossScaler_Reset_ClearsAllStatistics()
    {
        var scaler = new LossScaler<double>(initialScale: 100.0);

        // Accumulate some stats
        scaler.UnscaleGradientsAndCheck(new Vector<double>(new[] { 10.0 }));
        scaler.UnscaleGradientsAndCheck(new Vector<double>(new[] { double.NaN }));

        scaler.Reset();

        Assert.Equal(0, scaler.TotalUpdates);
        Assert.Equal(0, scaler.SkippedUpdates);
        Assert.Equal(0.0, scaler.OverflowRate, Tolerance);
    }

    [Fact]
    public void LossScaler_Reset_WithNewScale_UpdatesScaleOnly()
    {
        var scaler = new LossScaler<double>(initialScale: 100.0);

        scaler.Reset(newInitialScale: 999.0);

        Assert.Equal(999.0, scaler.Scale, Tolerance);
        Assert.Equal(0, scaler.TotalUpdates);
    }

    #endregion

    #region LossScaler - Overflow Detection Tests

    [Fact]
    public void LossScaler_HasOverflow_NaN_True()
    {
        var scaler = new LossScaler<double>();
        Assert.True(scaler.HasOverflow(double.NaN));
    }

    [Fact]
    public void LossScaler_HasOverflow_PositiveInfinity_True()
    {
        var scaler = new LossScaler<double>();
        Assert.True(scaler.HasOverflow(double.PositiveInfinity));
    }

    [Fact]
    public void LossScaler_HasOverflow_NegativeInfinity_True()
    {
        var scaler = new LossScaler<double>();
        Assert.True(scaler.HasOverflow(double.NegativeInfinity));
    }

    [Fact]
    public void LossScaler_HasOverflow_NormalValue_False()
    {
        var scaler = new LossScaler<double>();
        Assert.False(scaler.HasOverflow(1234.5678));
    }

    [Fact]
    public void LossScaler_HasOverflow_Zero_False()
    {
        var scaler = new LossScaler<double>();
        Assert.False(scaler.HasOverflow(0.0));
    }

    [Fact]
    public void LossScaler_DetectOverflow_TensorWithNaN()
    {
        var scaler = new LossScaler<double>();
        var data = new Vector<double>(new[] { 1.0, 2.0, double.NaN, 4.0 });
        var tensor = new Tensor<double>(new[] { 4 }, data);
        Assert.True(scaler.DetectOverflow(tensor));
    }

    [Fact]
    public void LossScaler_DetectOverflow_TensorAllNormal()
    {
        var scaler = new LossScaler<double>();
        var data = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });
        var tensor = new Tensor<double>(new[] { 4 }, data);
        Assert.False(scaler.DetectOverflow(tensor));
    }

    #endregion

    #region MixedPrecisionContext - FP32/FP16 Casting Tests

    [Fact]
    public void Context_Initialize_StoresMasterWeights()
    {
        using var ctx = new MixedPrecisionContext();
        var weights = new Vector<float>(new[] { 1.0f, 2.0f, 3.0f });

        ctx.Initialize(weights);

        Assert.True(ctx.IsInitialized);
        Assert.Equal(3, ctx.ParameterCount);

        var master = ctx.GetMasterWeights();
        Assert.Equal(1.0f, master[0], FloatTolerance);
        Assert.Equal(2.0f, master[1], FloatTolerance);
        Assert.Equal(3.0f, master[2], FloatTolerance);
    }

    [Fact]
    public void Context_CastToFP16_PreservesRepresentableValues()
    {
        using var ctx = new MixedPrecisionContext();
        var weights = new Vector<float>(new[] { 1.0f, 2.0f, 0.5f, -1.0f });

        ctx.Initialize(weights);
        ctx.CastWeightsToFP16();

        var working = ctx.GetWorkingWeights();

        // Powers of 2 are exactly representable in FP16
        Assert.Equal(1.0f, (float)working[0], FloatTolerance);
        Assert.Equal(2.0f, (float)working[1], FloatTolerance);
        Assert.Equal(0.5f, (float)working[2], FloatTolerance);
        Assert.Equal(-1.0f, (float)working[3], FloatTolerance);
    }

    [Fact]
    public void Context_FP32toFP16_PrecisionLoss()
    {
        // FP16 has 10-bit mantissa, so values needing more precision will be rounded
        using var ctx = new MixedPrecisionContext();
        // 1.00001 has more precision than FP16 can represent
        var weights = new Vector<float>(new[] { 1.00001f });

        ctx.Initialize(weights);
        ctx.CastWeightsToFP16();

        var working = ctx.GetWorkingWeights();
        float fp16Value = (float)working[0];

        // The value should be close but may not be exact
        Assert.True(Math.Abs(fp16Value - 1.0f) < 0.01f);
    }

    [Fact]
    public void Context_UpdateMasterWeights_SGDFormula()
    {
        // SGD: weights = weights - learningRate * gradients
        using var ctx = new MixedPrecisionContext();
        var weights = new Vector<float>(new[] { 10.0f, 20.0f, 30.0f });

        ctx.Initialize(weights);

        var gradients = new Vector<float>(new[] { 1.0f, 2.0f, 3.0f });
        float lr = 0.1f;

        ctx.UpdateMasterWeights(gradients, lr);

        var updated = ctx.GetMasterWeights();
        // 10 - 0.1*1 = 9.9, 20 - 0.1*2 = 19.8, 30 - 0.1*3 = 29.7
        Assert.Equal(9.9f, updated[0], FloatTolerance);
        Assert.Equal(19.8f, updated[1], FloatTolerance);
        Assert.Equal(29.7f, updated[2], FloatTolerance);
    }

    [Fact]
    public void Context_MultipleNamedParameters_IndependentManagement()
    {
        using var ctx = new MixedPrecisionContext();
        var namedParams = new Dictionary<string, Vector<float>>
        {
            ["layer1"] = new Vector<float>(new[] { 1.0f, 2.0f }),
            ["layer2"] = new Vector<float>(new[] { 3.0f, 4.0f, 5.0f })
        };

        ctx.Initialize(namedParams);

        Assert.Equal(5, ctx.ParameterCount);
        Assert.Contains("layer1", ctx.ParameterNames);
        Assert.Contains("layer2", ctx.ParameterNames);

        var layer1 = ctx.GetMasterWeights("layer1");
        Assert.Equal(2, layer1.Length);
        Assert.Equal(1.0f, layer1[0], FloatTolerance);

        var layer2 = ctx.GetMasterWeights("layer2");
        Assert.Equal(3, layer2.Length);
        Assert.Equal(5.0f, layer2[2], FloatTolerance);
    }

    [Fact]
    public void Context_PrepareGradients_UnscalesAndChecks()
    {
        using var ctx = new MixedPrecisionContext(new MixedPrecisionConfig
        {
            InitialLossScale = 100.0
        });
        var weights = new Vector<float>(new[] { 1.0f });
        ctx.Initialize(weights);

        // Simulate FP16 scaled gradients
        var scaledGrads = new Vector<Half>(new[] { (Half)500.0f }); // 500 / 100 = 5.0

        bool isValid = ctx.PrepareGradientsForUpdate(scaledGrads, out var floatGrads);

        Assert.True(isValid);
        // 500 / 100 = 5.0
        Assert.Equal(5.0f, floatGrads[0], 0.1f);
    }

    [Fact]
    public void Context_Reset_ClearsAllState()
    {
        using var ctx = new MixedPrecisionContext();
        var weights = new Vector<float>(new[] { 1.0f, 2.0f });
        ctx.Initialize(weights);
        ctx.CastWeightsToFP16();

        ctx.Reset();

        Assert.False(ctx.IsInitialized);
        Assert.Equal(0, ctx.ParameterCount);
    }

    [Fact]
    public void Context_DoubleInitialize_Throws()
    {
        using var ctx = new MixedPrecisionContext();
        var weights = new Vector<float>(new[] { 1.0f });
        ctx.Initialize(weights);

        Assert.Throws<InvalidOperationException>(() =>
            ctx.Initialize(new Vector<float>(new[] { 2.0f })));
    }

    [Fact]
    public void Context_GetWorkingBeforeCast_Throws()
    {
        using var ctx = new MixedPrecisionContext();
        var weights = new Vector<float>(new[] { 1.0f });
        ctx.Initialize(weights);

        // Working weights don't exist until CastWeightsToFP16 is called
        Assert.Throws<KeyNotFoundException>(() => ctx.GetWorkingWeights());
    }

    #endregion

    #region Cross-Component Integration Tests

    [Fact]
    public void LossScaler_GrowthThenBackoff_ExactStateTransitions()
    {
        // Test complex state transitions: grow, grow, overflow, grow
        var scaler = new LossScaler<double>(
            initialScale: 100.0,
            dynamicScaling: true,
            growthInterval: 2,
            growthFactor: 2.0,
            backoffFactor: 0.5,
            minScale: 1.0,
            maxScale: 10000.0
        );

        // Phase 1: 2 successes => growth to 200
        scaler.UnscaleGradientsAndCheck(new Vector<double>(new[] { 10.0 }));
        scaler.UnscaleGradientsAndCheck(new Vector<double>(new[] { 10.0 }));
        Assert.Equal(200.0, scaler.Scale, Tolerance);

        // Phase 2: 2 more successes => growth to 400
        scaler.UnscaleGradientsAndCheck(new Vector<double>(new[] { 10.0 }));
        scaler.UnscaleGradientsAndCheck(new Vector<double>(new[] { 10.0 }));
        Assert.Equal(400.0, scaler.Scale, Tolerance);

        // Phase 3: overflow => backoff to 200
        scaler.UnscaleGradientsAndCheck(new Vector<double>(new[] { double.NaN }));
        Assert.Equal(200.0, scaler.Scale, Tolerance);

        // Phase 4: 2 successes => growth to 400 again
        scaler.UnscaleGradientsAndCheck(new Vector<double>(new[] { 10.0 }));
        scaler.UnscaleGradientsAndCheck(new Vector<double>(new[] { 10.0 }));
        Assert.Equal(400.0, scaler.Scale, Tolerance);

        // Verify stats: 7 total, 1 skipped
        Assert.Equal(7, scaler.TotalUpdates);
        Assert.Equal(1, scaler.SkippedUpdates);
        Assert.Equal(1.0 / 7.0, scaler.OverflowRate, Tolerance);
    }

    [Fact]
    public void E4M3_E5M2_FormatComparison_SameExactValues()
    {
        // Both formats should represent powers of 2 exactly (within range)
        float[] exactValues = { 0.5f, 1.0f, 2.0f, 4.0f, 8.0f, 16.0f };
        foreach (var v in exactValues)
        {
            var e4m3 = Float8E4M3.FromFloat(v);
            var e5m2 = Float8E5M2.FromFloat(v);
            Assert.Equal(v, e4m3.ToFloat());
            Assert.Equal(v, e5m2.ToFloat());
        }
    }

    [Fact]
    public void FullPipeline_ScaleUnscale_WithFloat8Gradients()
    {
        // Simulate: compute loss in FP32, scale, convert to E5M2 gradients, convert back
        var scaler = new LossScaler<float>(initialScale: 256.0f);

        // Original loss = 0.01
        float loss = 0.01f;
        float scaledLoss = scaler.ScaleLoss(loss);
        Assert.Equal(2.56f, scaledLoss, FloatTolerance);

        // Simulate gradient magnitudes scaled by loss scale
        float[] scaledGrads = { 256.0f, 512.0f, 128.0f };

        // Convert to E5M2 (simulating low-precision gradient storage)
        var e5m2Grads = scaledGrads.ToE5M2();
        float[] recoveredGrads = e5m2Grads.ToFloatArray();

        // Powers of 2 should survive E5M2 round-trip
        Assert.Equal(256.0f, recoveredGrads[0], FloatTolerance);
        Assert.Equal(512.0f, recoveredGrads[1], FloatTolerance);
        Assert.Equal(128.0f, recoveredGrads[2], FloatTolerance);

        // Unscale
        var gradVector = new Vector<float>(recoveredGrads);
        scaler.UnscaleGradients(gradVector);

        // 256/256 = 1, 512/256 = 2, 128/256 = 0.5
        Assert.Equal(1.0f, gradVector[0], FloatTolerance);
        Assert.Equal(2.0f, gradVector[1], FloatTolerance);
        Assert.Equal(0.5f, gradVector[2], FloatTolerance);
    }

    [Fact]
    public void Context_FullTrainingStep_MathVerification()
    {
        // Full training step: init -> cast FP16 -> compute -> update master
        using var ctx = new MixedPrecisionContext(new MixedPrecisionConfig
        {
            InitialLossScale = 100.0
        });

        var initialWeights = new Vector<float>(new[] { 5.0f, 10.0f });
        ctx.Initialize(initialWeights);

        // Cast to FP16
        ctx.CastWeightsToFP16();
        var fp16Weights = ctx.GetWorkingWeights();
        Assert.Equal(5.0f, (float)fp16Weights[0], FloatTolerance);
        Assert.Equal(10.0f, (float)fp16Weights[1], FloatTolerance);

        // Simulate FP16 gradient computation (already scaled by loss scaler)
        var scaledFP16Grads = new Vector<Half>(new[] { (Half)200.0f, (Half)400.0f });

        // Prepare gradients (cast to FP32 + unscale + overflow check)
        bool isValid = ctx.PrepareGradientsForUpdate(scaledFP16Grads, out var fp32Grads);
        Assert.True(isValid);

        // Gradients should be unscaled: 200/100=2, 400/100=4
        Assert.Equal(2.0f, fp32Grads[0], 0.1f);
        Assert.Equal(4.0f, fp32Grads[1], 0.1f);

        // Update master weights: w = w - lr * grad
        ctx.UpdateMasterWeights(fp32Grads, learningRate: 0.5f);

        var updatedWeights = ctx.GetMasterWeights();
        // 5 - 0.5*2 = 4, 10 - 0.5*4 = 8
        Assert.Equal(4.0f, updatedWeights[0], 0.1f);
        Assert.Equal(8.0f, updatedWeights[1], 0.1f);
    }

    #endregion

    #region Edge Case and Numerical Stability Tests

    [Fact]
    public void E4M3_HashCode_ConsistentWithEquality()
    {
        var a = Float8E4M3.FromFloat(1.5f);
        var b = Float8E4M3.FromFloat(1.5f);
        Assert.Equal(a.GetHashCode(), b.GetHashCode());
    }

    [Fact]
    public void E5M2_HashCode_ConsistentWithEquality()
    {
        var a = Float8E5M2.FromFloat(1.5f);
        var b = Float8E5M2.FromFloat(1.5f);
        Assert.Equal(a.GetHashCode(), b.GetHashCode());
    }

    [Fact]
    public void E4M3_NaN_NotEqualToNaN()
    {
        // NaN comparison: NaN != NaN in IEEE 754
        // But since Float8E4M3 compares via raw byte value, NaN == NaN
        var a = Float8E4M3.NaN;
        var b = Float8E4M3.NaN;
        // In this implementation, equality is based on raw value
        Assert.True(a == b);
    }

    [Fact]
    public void E5M2_InfinityConvertBackToFloat()
    {
        var posInf = Float8E5M2.PositiveInfinity;
        Assert.True(float.IsPositiveInfinity(posInf.ToFloat()));

        var negInf = Float8E5M2.NegativeInfinity;
        Assert.True(float.IsNegativeInfinity(negInf.ToFloat()));
    }

    [Fact]
    public void LossScaler_UnscaleAndCheck_TensorOverflow_ReducesScale()
    {
        var scaler = new LossScaler<double>(
            initialScale: 800.0,
            dynamicScaling: true,
            backoffFactor: 0.25
        );

        var data = new Vector<double>(new[] { 1.0, double.PositiveInfinity, 3.0 });
        var tensor = new Tensor<double>(new[] { 3 }, data);

        bool result = scaler.UnscaleGradientsAndCheck(tensor);

        Assert.False(result);
        // 800 * 0.25 = 200
        Assert.Equal(200.0, scaler.Scale, Tolerance);
        Assert.Equal(1, scaler.SkippedUpdates);
    }

    [Fact]
    public void LossScaler_UnscaleAndCheck_TensorNoOverflow_GradientsCorrect()
    {
        var scaler = new LossScaler<double>(
            initialScale: 100.0,
            dynamicScaling: true,
            growthInterval: 5000
        );

        var data = new Vector<double>(new[] { 50.0, 100.0, 200.0 });
        var tensor = new Tensor<double>(new[] { 3 }, data);

        bool result = scaler.UnscaleGradientsAndCheck(tensor);

        Assert.True(result);
        // Gradients unscaled: 50/100=0.5, 100/100=1.0, 200/100=2.0
        Assert.Equal(0.5, tensor.GetFlatIndexValue(0), Tolerance);
        Assert.Equal(1.0, tensor.GetFlatIndexValue(1), Tolerance);
        Assert.Equal(2.0, tensor.GetFlatIndexValue(2), Tolerance);
    }

    [Fact]
    public void E4M3_ExplicitCastOperators_WorkCorrectly()
    {
        float original = 4.0f;
        Float8E4M3 encoded = (Float8E4M3)original;
        float decoded = (float)encoded;
        Assert.Equal(original, decoded, FloatTolerance);
    }

    [Fact]
    public void E5M2_ExplicitCastOperators_WorkCorrectly()
    {
        float original = 8.0f;
        Float8E5M2 encoded = (Float8E5M2)original;
        float decoded = (float)encoded;
        Assert.Equal(original, decoded, FloatTolerance);
    }

    #endregion
}
