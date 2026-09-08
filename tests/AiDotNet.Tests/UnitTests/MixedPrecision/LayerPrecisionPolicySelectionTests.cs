using AiDotNet.Enums;
using AiDotNet.MixedPrecision;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MixedPrecision;

/// <summary>
/// Covers <see cref="LayerPrecisionPolicy.ForPrecision"/>, which picks the per-layer policy that goes
/// with a precision type.
/// </summary>
/// <remarks>
/// <para>
/// This mapping was the only live logic inside <c>MixedPrecisionTrainingLoop</c> — a public type nothing
/// constructed, whose training counters were advanced by nothing and whose fields were assigned and
/// never read. The type was removed and the mapping promoted here, beside the per-precision factories it
/// chooses between (#2099).
/// </para>
/// </remarks>
public class LayerPrecisionPolicySelectionTests
{
    [Theory]
    [InlineData(MixedPrecisionType.FP16)]
    [InlineData(MixedPrecisionType.BF16)]
    [InlineData(MixedPrecisionType.FP8_E4M3)]
    [InlineData(MixedPrecisionType.FP8_E5M2)]
    [InlineData(MixedPrecisionType.FP8_Hybrid)]
    public void EveryPrecisionTypeSelectsItsOwnDefaultPrecision(MixedPrecisionType precisionType)
    {
        var policy = LayerPrecisionPolicy.ForPrecision(precisionType);

        Assert.NotNull(policy);
        Assert.Equal(Expected(precisionType), policy.DefaultPrecision);
    }

    /// <summary>
    /// The three FP8 formats share one policy: they differ in numeric range, not in which layers can
    /// survive being cast to them.
    /// </summary>
    [Fact]
    public void TheFp8FormatsShareTheFp8Policy()
    {
        var expected = LayerPrecisionPolicy.ForFP8().DefaultPrecision;

        Assert.Equal(expected, LayerPrecisionPolicy.ForPrecision(MixedPrecisionType.FP8_E4M3).DefaultPrecision);
        Assert.Equal(expected, LayerPrecisionPolicy.ForPrecision(MixedPrecisionType.FP8_E5M2).DefaultPrecision);
        Assert.Equal(expected, LayerPrecisionPolicy.ForPrecision(MixedPrecisionType.FP8_Hybrid).DefaultPrecision);
    }

    /// <summary>
    /// The selection has to agree with the factory it selects, or callers get a different policy
    /// depending on which door they came through.
    /// </summary>
    [Fact]
    public void SelectingAgreesWithCallingTheFactoryDirectly()
    {
        Assert.Equal(
            LayerPrecisionPolicy.ForFP16().DefaultPrecision,
            LayerPrecisionPolicy.ForPrecision(MixedPrecisionType.FP16).DefaultPrecision);

        Assert.Equal(
            LayerPrecisionPolicy.ForBF16().DefaultPrecision,
            LayerPrecisionPolicy.ForPrecision(MixedPrecisionType.BF16).DefaultPrecision);
    }

    /// <summary>
    /// Normalization layers are the ones that do not tolerate coarse precision, so an FP16 policy that
    /// let them be cast down would be the policy failing at its one job.
    /// </summary>
    [Fact]
    public void TheFp16PolicyKeepsNormalizationLayersInFp32()
    {
        var policy = LayerPrecisionPolicy.ForPrecision(MixedPrecisionType.FP16);

        // The enum has no FP32 member; "leave this layer alone" is MixedPrecisionType.None, which is
        // what KeepInFP32 records.
        Assert.Equal(MixedPrecisionType.None, policy.GetPrecision("BatchNorm"));
        Assert.Equal(MixedPrecisionType.None, policy.GetPrecision("LayerNorm"));
        Assert.Equal(MixedPrecisionType.None, policy.GetPrecision("RMSNorm"));

        // An ordinary layer is left at the policy's default rather than pinned up.
        Assert.Equal(MixedPrecisionType.FP16, policy.GetPrecision("Dense"));
    }

    /// <summary>
    /// A config names a precision, so the mapping has to accept one straight from it — that is the case
    /// the removed type existed to serve.
    /// </summary>
    [Fact]
    public void ItTakesAPrecisionStraightFromAConfig()
    {
        var config = new MixedPrecisionConfig { PrecisionType = MixedPrecisionType.BF16 };

        var policy = LayerPrecisionPolicy.ForPrecision(config.PrecisionType);

        Assert.Equal(MixedPrecisionType.BF16, policy.DefaultPrecision);
    }

    private static MixedPrecisionType Expected(MixedPrecisionType precisionType) => precisionType switch
    {
        MixedPrecisionType.BF16 => MixedPrecisionType.BF16,
        MixedPrecisionType.FP8_E4M3 or
        MixedPrecisionType.FP8_E5M2 or
        MixedPrecisionType.FP8_Hybrid => LayerPrecisionPolicy.ForFP8().DefaultPrecision,
        _ => MixedPrecisionType.FP16
    };
}
