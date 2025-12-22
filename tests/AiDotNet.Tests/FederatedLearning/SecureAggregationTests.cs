using AiDotNet.FederatedLearning.Privacy;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

public class SecureAggregationTests
{
    [Fact]
    public void AggregateSumSecurely_CancelsPairwiseMasks()
    {
        var secureAggregation = new SecureAggregation<double>(parameterCount: 3, randomSeed: 123);
        secureAggregation.GeneratePairwiseSecrets(new List<int> { 1, 2, 3 });

        Assert.Equal(3, secureAggregation.GetClientCount());

        var update1 = new Dictionary<string, double[]>
        {
            ["a"] = new[] { 1.0, 2.0 },
            ["b"] = new[] { 3.0 }
        };
        var update2 = new Dictionary<string, double[]>
        {
            ["a"] = new[] { -1.0, 4.0 },
            ["b"] = new[] { 1.0 }
        };
        var update3 = new Dictionary<string, double[]>
        {
            ["a"] = new[] { 0.5, -0.25 },
            ["b"] = new[] { 2.0 }
        };

        var maskedUpdates = new Dictionary<int, Dictionary<string, double[]>>
        {
            [1] = secureAggregation.MaskUpdate(1, update1),
            [2] = secureAggregation.MaskUpdate(2, update2),
            [3] = secureAggregation.MaskUpdate(3, update3)
        };

        var aggregatedSum = secureAggregation.AggregateSumSecurely(maskedUpdates);

        Assert.Equal(2, aggregatedSum["a"].Length);
        Assert.Equal(1, aggregatedSum["b"].Length);

        Assert.Equal(update1["a"][0] + update2["a"][0] + update3["a"][0], aggregatedSum["a"][0], precision: 10);
        Assert.Equal(update1["a"][1] + update2["a"][1] + update3["a"][1], aggregatedSum["a"][1], precision: 10);
        Assert.Equal(update1["b"][0] + update2["b"][0] + update3["b"][0], aggregatedSum["b"][0], precision: 10);

        secureAggregation.ClearSecrets();
        Assert.Equal(0, secureAggregation.GetClientCount());
    }

    [Fact]
    public void AggregateSecurely_ReturnsWeightedAverage_WhenClientsPreWeightUpdates()
    {
        var secureAggregation = new SecureAggregation<double>(parameterCount: 2, randomSeed: 42);
        secureAggregation.GeneratePairwiseSecrets(new List<int> { 10, 20 });

        var update10 = new Dictionary<string, double[]>
        {
            ["w"] = new[] { 1.0, 2.0 }
        };
        var update20 = new Dictionary<string, double[]>
        {
            ["w"] = new[] { 3.0, 4.0 }
        };

        var weights = new Dictionary<int, double>
        {
            [10] = 1.0,
            [20] = 3.0
        };

        var maskedUpdates = new Dictionary<int, Dictionary<string, double[]>>
        {
            [10] = secureAggregation.MaskUpdate(10, update10, weights[10]),
            [20] = secureAggregation.MaskUpdate(20, update20, weights[20])
        };

        var aggregated = secureAggregation.AggregateSecurely(maskedUpdates, weights);

        Assert.Single(aggregated);
        Assert.True(aggregated.TryGetValue("w", out var parameters));
        Assert.Equal(2, parameters.Length);

        var expected0 = (update10["w"][0] * weights[10] + update20["w"][0] * weights[20]) / (weights[10] + weights[20]);
        var expected1 = (update10["w"][1] * weights[10] + update20["w"][1] * weights[20]) / (weights[10] + weights[20]);

        Assert.Equal(expected0, parameters[0], precision: 10);
        Assert.Equal(expected1, parameters[1], precision: 10);
    }

    [Fact]
    public void GeneratePairwiseSecrets_ThrowsForInvalidClientLists()
    {
        var secureAggregation = new SecureAggregation<double>(parameterCount: 1, randomSeed: 123);

        Assert.Throws<ArgumentException>(() => secureAggregation.GeneratePairwiseSecrets(null!));
        Assert.Throws<ArgumentException>(() => secureAggregation.GeneratePairwiseSecrets(new List<int> { 1 }));
    }

    [Fact]
    public void MaskUpdate_ThrowsForInvalidInputs()
    {
        var secureAggregation = new SecureAggregation<double>(parameterCount: 2, randomSeed: 7);

        Assert.Throws<ArgumentException>(() => secureAggregation.MaskUpdate(1, new Dictionary<string, double[]>()));
        Assert.Throws<ArgumentException>(() => secureAggregation.MaskUpdate(1, new Dictionary<string, double[]> { ["w"] = new[] { 1.0, 2.0 } }));
        Assert.Throws<ArgumentException>(() => secureAggregation.MaskUpdate(1, new Dictionary<string, double[]> { ["w"] = new[] { 1.0, 2.0 } }, clientWeight: 0.0));

        secureAggregation.GeneratePairwiseSecrets(new List<int> { 1, 2 });

        Assert.Throws<ArgumentException>(() => secureAggregation.MaskUpdate(1, new Dictionary<string, double[]> { ["w"] = new[] { 1.0, 2.0, 3.0 } }));
    }

    [Fact]
    public void AggregateSecurely_ThrowsWhenWeightsMissingOrInvalid()
    {
        var secureAggregation = new SecureAggregation<double>(parameterCount: 1, randomSeed: 11);
        secureAggregation.GeneratePairwiseSecrets(new List<int> { 1, 2 });

        var update1 = new Dictionary<string, double[]> { ["w"] = new[] { 1.0 } };
        var update2 = new Dictionary<string, double[]> { ["w"] = new[] { 2.0 } };

        var maskedUpdates = new Dictionary<int, Dictionary<string, double[]>>
        {
            [1] = secureAggregation.MaskUpdate(1, update1, clientWeight: 1.0),
            [2] = secureAggregation.MaskUpdate(2, update2, clientWeight: 1.0)
        };

        Assert.Throws<ArgumentException>(() => secureAggregation.AggregateSecurely(maskedUpdates, clientWeights: new Dictionary<int, double>()));
        Assert.Throws<ArgumentException>(() => secureAggregation.AggregateSecurely(maskedUpdates, clientWeights: new Dictionary<int, double> { [1] = 1.0 }));
        Assert.Throws<ArgumentException>(() => secureAggregation.AggregateSecurely(maskedUpdates, clientWeights: new Dictionary<int, double> { [1] = -1.0, [2] = 0.0 }));
    }

    [Fact]
    public void Dispose_MakesInstanceUnusable()
    {
        var secureAggregation = new SecureAggregation<double>(parameterCount: 1, randomSeed: 1);
        secureAggregation.Dispose();

        Assert.Throws<ObjectDisposedException>(() => secureAggregation.ClearSecrets());
    }
}
