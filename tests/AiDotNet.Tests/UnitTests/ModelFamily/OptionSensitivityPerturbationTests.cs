using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ModelFamily;

public sealed class OptionSensitivityPerturbationTests
{
    [Fact]
    public void ScalePerturbationTriesTheSmallerDivisorBeforeCoprimePadding()
    {
        Assert.Equal(2, OptionSensitivityHarness.Perturbations(typeof(int), 4, "ScaleFactor").First());
    }

    [Theory]
    [InlineData("CodecRates")]
    [InlineData("CodecRatios")]
    [InlineData("UpsampleRates")]
    [InlineData("UpsampleStrides")]
    [InlineData("BlockStrides")]
    public void CoupledStageGeometryHasADifferentCandidateWithTheSameHop(string option)
    {
        // SeedTTS fixes the codec hop to the product of [2,2,2]. Single-element and length
        // changes alter that hop; equal factors cannot be tested by reversing the array either.
        int[] original = { 2, 2, 2 };
        var candidates = OptionSensitivityHarness.Perturbations(typeof(int[]), original, option).Cast<int[]>();
        Assert.Contains(candidates, candidate => candidate.All(rate => rate >= 2)
            && candidate.Aggregate(1, (product, rate) => product * rate) == 8
            && !candidate.SequenceEqual(original));
        Assert.Equal(new[] { 2, 2, 2 }, original);
    }
}
