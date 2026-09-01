using AiDotNet.Enums;
using AiDotNet.Evolution;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution;

public sealed class EvolutionDescriptorTests
{
    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void NonFiniteValuesAreAlwaysRejected(double value)
    {
        foreach (EvolutionOutOfRangePolicy policy in Enum.GetValues(typeof(EvolutionOutOfRangePolicy)))
        {
            var descriptor = new EvolutionDescriptorDefinition("x", 0, 1, 4, policy);
            Assert.False(descriptor.TryGetBin(value, out _));
        }
    }

    [Fact]
    public void ExactBoundsAndAdjacentValuesFollowExplicitPolicies()
    {
        var reject = new EvolutionDescriptorDefinition("x", 0, 1, 4, EvolutionOutOfRangePolicy.Reject);
        Assert.True(reject.TryGetBin(0, out int exactMin));
        Assert.True(reject.TryGetBin(1, out int exactMax));
        Assert.False(reject.TryGetBin(-double.Epsilon, out _));
        Assert.False(reject.TryGetBin(1.0000000000000002, out _));
        Assert.Equal(0, exactMin);
        Assert.Equal(3, exactMax);

        var clamp = new EvolutionDescriptorDefinition("x", 0, 1, 4, EvolutionOutOfRangePolicy.Clamp);
        Assert.True(clamp.TryGetBin(-1, out int below));
        Assert.True(clamp.TryGetBin(2, out int above));
        Assert.Equal(0, below);
        Assert.Equal(3, above);

        var overflow = new EvolutionDescriptorDefinition("x", 0, 1, 4, EvolutionOutOfRangePolicy.OverflowBins);
        Assert.Equal(6, overflow.EffectiveBinCount);
        Assert.True(overflow.TryGetBin(-1, out below));
        Assert.True(overflow.TryGetBin(0, out exactMin));
        Assert.True(overflow.TryGetBin(1, out exactMax));
        Assert.True(overflow.TryGetBin(2, out above));
        Assert.Equal(0, below);
        Assert.Equal(1, exactMin);
        Assert.Equal(4, exactMax);
        Assert.Equal(5, above);
    }

    [Fact]
    public void CalibratorIsOrderIndependentAndFreezesConstantData()
    {
        var first = new EvolutionDescriptorCalibrator("x", 8);
        var second = new EvolutionDescriptorCalibrator("x", 8);
        foreach (double value in new[] { 4.0, -2.0, 7.0, 1.0 }) first.Observe(value);
        foreach (double value in new[] { 1.0, 7.0, -2.0, 4.0 }) second.Observe(value);

        Assert.Equal(first.Freeze(0.1).ToCanonicalString(), second.Freeze(0.1).ToCanonicalString());

        var constant = new EvolutionDescriptorCalibrator("constant", 2);
        constant.Observe(5);
        EvolutionDescriptorDefinition definition = constant.Freeze();
        Assert.True(definition.Maximum > definition.Minimum);
        Assert.True(definition.TryGetBin(5, out _));
    }

    [Fact]
    public void CalibratorHandlesExtremeConstantsAndRejectsUnrepresentableSpan()
    {
        foreach (double value in new[] { -double.MaxValue, 0.0, double.MaxValue })
        {
            var calibrator = new EvolutionDescriptorCalibrator("extreme", 4);
            calibrator.Observe(value);
            EvolutionDescriptorDefinition definition = calibrator.Freeze();

            Assert.True(definition.Minimum < definition.Maximum);
            Assert.True(definition.TryGetBin(value, out _));
        }

        var unrepresentable = new EvolutionDescriptorCalibrator("wide", 4);
        unrepresentable.Observe(-double.MaxValue);
        unrepresentable.Observe(double.MaxValue);
        Assert.Throws<InvalidOperationException>(() => unrepresentable.Freeze());
    }

    [Fact]
    public void InvalidDefinitionsAreRejectedBeforeArchiveAllocation()
    {
        Assert.Throws<ArgumentException>(() => new EvolutionDescriptorDefinition("", 0, 1, 2));
        Assert.Throws<ArgumentException>(() => new EvolutionDescriptorDefinition("x", 1, 1, 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => new EvolutionDescriptorDefinition("x", double.NaN, 1, 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => new EvolutionDescriptorDefinition("x", 0, 1, 0));
        Assert.Throws<ArgumentException>(() => new MapElitesArchive<TestGenome>(Array.Empty<EvolutionDescriptorDefinition>()));
        Assert.Throws<ArgumentException>(() => new MapElitesArchive<TestGenome>(new[]
        {
            new EvolutionDescriptorDefinition("x", 0, 1, 2),
            new EvolutionDescriptorDefinition("x", 0, 1, 2)
        }));
    }
}
