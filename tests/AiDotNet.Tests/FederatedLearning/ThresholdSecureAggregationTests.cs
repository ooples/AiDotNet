using AiDotNet.FederatedLearning.Privacy;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

public class ThresholdSecureAggregationTests
{
    [Fact]
    public void MasksCancel_ForStructuredUpdates()
    {
        var secureAggregation = new ThresholdSecureAggregation<double>(parameterCount: 3, randomSeed: 123);
        secureAggregation.InitializeRound(new List<int> { 0, 1, 2 });

        var update0 = new Dictionary<string, double[]>
        {
            ["a"] = new[] { 1.0 },
            ["b"] = new[] { 2.0, 3.0 }
        };

        var update1 = new Dictionary<string, double[]>
        {
            ["a"] = new[] { 4.0 },
            ["b"] = new[] { 5.0, 6.0 }
        };

        var update2 = new Dictionary<string, double[]>
        {
            ["a"] = new[] { 7.0 },
            ["b"] = new[] { 8.0, 9.0 }
        };

        var maskedUpdates = new Dictionary<int, Dictionary<string, double[]>>
        {
            [0] = secureAggregation.MaskUpdate(0, update0),
            [1] = secureAggregation.MaskUpdate(1, update1),
            [2] = secureAggregation.MaskUpdate(2, update2)
        };

        var sum = secureAggregation.AggregateSumSecurely(maskedUpdates);

        Assert.Equal(12.0, sum["a"][0], precision: 12);
        Assert.Equal(15.0, sum["b"][0], precision: 12);
        Assert.Equal(18.0, sum["b"][1], precision: 12);
    }

    [Fact]
    public void AggregateSumSecurely_Succeeds_WithUnmaskingDropout_ForStructuredUpdates()
    {
        var secureAggregation = new ThresholdSecureAggregation<double>(parameterCount: 3, randomSeed: 42);
        secureAggregation.InitializeRound(new List<int> { 0, 1, 2 });

        var update0 = new Dictionary<string, double[]>
        {
            ["a"] = new[] { 1.0 },
            ["b"] = new[] { 2.0, 3.0 }
        };

        var update1 = new Dictionary<string, double[]>
        {
            ["a"] = new[] { 4.0 },
            ["b"] = new[] { 5.0, 6.0 }
        };

        var update2 = new Dictionary<string, double[]>
        {
            ["a"] = new[] { 7.0 },
            ["b"] = new[] { 8.0, 9.0 }
        };

        var maskedUpdates = new Dictionary<int, Dictionary<string, double[]>>
        {
            [0] = secureAggregation.MaskUpdate(0, update0),
            [1] = secureAggregation.MaskUpdate(1, update1),
            [2] = secureAggregation.MaskUpdate(2, update2)
        };

        var sum = secureAggregation.AggregateSumSecurely(maskedUpdates, unmaskingClientIds: new[] { 0, 1 });

        Assert.Equal(12.0, sum["a"][0], precision: 12);
        Assert.Equal(15.0, sum["b"][0], precision: 12);
        Assert.Equal(18.0, sum["b"][1], precision: 12);
    }
}

