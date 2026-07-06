#if NET7_0_OR_GREATER
using System;
using AiDotNet.MixedPrecision;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Optimizers;

/// <summary>
/// Proves the SIMD bulk BF16 quantize/dequantize path in <see cref="Adam8BitOptimizer{T, TInput, TOutput}"/>
/// is bit-identical to the scalar <see cref="BitConverterHelper.FloatToBf16Bits(float)"/> /
/// <see cref="BitConverterHelper.Bf16BitsToFloat(ushort)"/> reference, over normals, zeros, subnormals,
/// infinities, NaN, round-to-nearest-even carry cases, and arrays whose length is not a multiple of the
/// hardware vector width (so the scalar tail is exercised too). The SIMD path is a NET7+ perf optimization
/// on the Adam8Bit BF16 moment-storage step; this test is the correctness gate for it.
/// </summary>
public class Adam8BitBf16SimdParityTests
{
    private static void AssertParity(float[] src)
    {
        var bf16 = new ushort[src.Length];
        var back = new float[src.Length];

        // Production path (SIMD block + scalar tail).
        Adam8BitOptimizer<float, Matrix<float>, Vector<float>>.Bf16RoundTripForTest(src, bf16, back);

        for (int i = 0; i < src.Length; i++)
        {
            // Quantize must match the scalar reference exactly (raw 16-bit pattern).
            ushort expectedBits = BitConverterHelper.FloatToBf16Bits(src[i]);
            Assert.True(expectedBits == bf16[i],
                $"float→bf16 mismatch at {i} for {src[i]} (0x{BitConverter.SingleToUInt32Bits(src[i]):X8}): " +
                $"expected 0x{expectedBits:X4}, got 0x{bf16[i]:X4}");

            // Dequantize must match the scalar reference exactly (compare bit patterns so NaN == NaN).
            float expectedBack = BitConverterHelper.Bf16BitsToFloat(expectedBits);
            uint expBackBits = BitConverter.SingleToUInt32Bits(expectedBack);
            uint actBackBits = BitConverter.SingleToUInt32Bits(back[i]);
            Assert.True(expBackBits == actBackBits,
                $"bf16→float mismatch at {i}: expected 0x{expBackBits:X8}, got 0x{actBackBits:X8}");
        }
    }

    [Fact]
    public void Bf16Simd_MatchesScalar_OnEdgeCases()
    {
        var edge = new float[]
        {
            0f, -0f, 1f, -1f, 2f, -2f, 0.5f, -0.5f,
            float.Epsilon, -float.Epsilon,                 // smallest subnormal
            1.17549435e-38f, -1.17549435e-38f,             // ~smallest normal
            float.MaxValue, float.MinValue,
            float.PositiveInfinity, float.NegativeInfinity,
            float.NaN, -float.NaN,
            BitConverter.UInt32BitsToSingle(0x7FC00001u),  // a specific quiet NaN payload
            BitConverter.UInt32BitsToSingle(0x7F800001u),  // a signaling NaN
            BitConverter.UInt32BitsToSingle(0x0000FFFFu),  // low-mantissa subnormal -> rounds toward 0
            BitConverter.UInt32BitsToSingle(0x3F7FFFFFu),  // just below 1.0, forces RNE carry
            BitConverter.UInt32BitsToSingle(0x3F7F8000u),  // exact bf16 boundary (lsb 0, no round)
            BitConverter.UInt32BitsToSingle(0x3F7F8001u),  // round-up case
            3.14159265f, -2.71828182f, 1e30f, -1e-30f,
        };
        AssertParity(edge);
    }

    [Theory]
    // Lengths spanning < one vector, exactly one+ vector, and non-multiples (forces the scalar tail).
    [InlineData(1, 11)]
    [InlineData(7, 22)]
    [InlineData(16, 33)]
    [InlineData(31, 44)]
    [InlineData(257, 55)]
    [InlineData(8193, 66)]   // > Bf16BulkChunkGrain, multiple parallel chunks
    [InlineData(20001, 77)]
    public void Bf16Simd_MatchesScalar_OnRandomArrays(int length, int seed)
    {
        var rng = new Random(seed);
        var src = new float[length];
        for (int i = 0; i < length; i++)
        {
            // Wide dynamic range, mixed sign, including some exact halves to hit RNE.
            double mag = Math.Pow(10, rng.NextDouble() * 60 - 30);
            float v = (float)(mag * (rng.Next(2) == 0 ? 1 : -1));
            src[i] = rng.Next(20) == 0 ? 0f : v;
        }
        AssertParity(src);
    }
}
#endif
