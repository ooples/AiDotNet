using System;
using System.Linq;
using AiDotNet.Optimizers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// The 8-bit dynamic codebook of Dettmers et al. (bitsandbytes <c>create_dynamic_map</c>) that Adam8Bit stores its
/// moments in: its structure, nearest-entry encoding, saturation, and refusal of NaN.
/// </summary>
public class DynamicQuantizationMapTests
{
    [Fact]
    public void Codebooks_HaveTheReferenceStructure()
    {
        foreach (var code in new[] { DynamicQuantizationMap.Signed, DynamicQuantizationMap.Unsigned })
        {
            Assert.Equal(256, code.Length);
            Assert.True(code.Zip(code.Skip(1), (a, b) => a < b).All(increasing => increasing), "entries must be strictly increasing");
            Assert.Contains(0.0, code);
            Assert.Equal(1.0, code[code.Length - 1]);
        }

        // Signed: every non-zero entry below 1.0 has its negation (1.0 has none: 127 positive means, 127 negated, 0, 1).
        var signed = DynamicQuantizationMap.Signed;
        foreach (double entry in signed.Where(e => e > 0.0 && e < 1.0))
            Assert.Contains(-entry, signed);
        Assert.Equal(0.0, signed[DynamicQuantizationMap.SignedZeroIndex]);

        // Unsigned: no negative entries.
        Assert.True(DynamicQuantizationMap.Unsigned.All(e => e >= 0.0));
    }

    [Fact]
    public void Encode_ReturnsTheNearestEntry_AndDecodeRestoresIt()
    {
        var code = DynamicQuantizationMap.Signed;
        const double scale = 2.0;
        for (int i = 1; i < code.Length - 1; i++)
        {
            // Each exact entry encodes to itself.
            Assert.Equal(i, DynamicQuantizationMap.Encode(code[i] * scale, scale, code));
            Assert.Equal(code[i] * scale, DynamicQuantizationMap.Decode((byte)i, scale, code));

            // A value a quarter of the way to the next entry still encodes to this one.
            double nearer = code[i] + (code[i + 1] - code[i]) * 0.25;
            Assert.Equal(i, DynamicQuantizationMap.Encode(nearer * scale, scale, code));
        }
    }

    [Fact]
    public void Encode_SaturatesOutOfRangeValues()
    {
        var code = DynamicQuantizationMap.Signed;
        Assert.Equal(code.Length - 1, DynamicQuantizationMap.Encode(double.PositiveInfinity, 1.0, code));
        Assert.Equal(0, DynamicQuantizationMap.Encode(double.NegativeInfinity, 1.0, code));
        Assert.Equal(code.Length - 1, DynamicQuantizationMap.Encode(5.0, 1.0, code));
    }

    [Fact]
    public void Encode_RefusesNaN_InsteadOfIndexingOutsideTheCodebook()
    {
        // NaN fails every comparison, so without a guard it skipped both clamps and read code[-1].
        foreach (var code in new[] { DynamicQuantizationMap.Signed, DynamicQuantizationMap.Unsigned })
        {
            var nearest = Assert.Throws<ArgumentException>(() => DynamicQuantizationMap.Encode(double.NaN, 1.0, code));
            Assert.Contains("NaN", nearest.Message);
            var stochastic = Assert.Throws<ArgumentException>(
                () => DynamicQuantizationMap.EncodeStochastic(double.NaN, 1.0, code, 0.5));
            Assert.Contains("NaN", stochastic.Message);
        }
    }
}