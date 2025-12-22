using AiDotNet.FederatedLearning.Selection;
using AiDotNet.Models;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

public class ClientSelectionStrategyTests
{
    [Fact]
    public void StratifiedSelection_ReducesGroupAllocations_WhenTooManyGroups()
    {
        var strategy = new StratifiedClientSelectionStrategy();

        var request = new ClientSelectionRequest
        {
            CandidateClientIds = new[] { 1, 2, 3, 4 },
            FractionToSelect = 0.5,
            Random = RandomHelper.CreateSeededRandom(123),
            ClientGroupKeys = new Dictionary<int, string>
            {
                [1] = "g1",
                [2] = "g2",
                [3] = "g3",
                [4] = "g4"
            }
        };

        var selected = strategy.SelectClients(request);

        Assert.Equal("Stratified", strategy.GetStrategyName());
        Assert.Equal(2, selected.Count);
        Assert.All(selected, id => Assert.Contains(id, request.CandidateClientIds));

        var groupKeys = request.ClientGroupKeys!;
        var selectedGroups = selected.Select(id => groupKeys[id]).Distinct(StringComparer.OrdinalIgnoreCase).ToList();
        Assert.Equal(selected.Count, selectedGroups.Count);
    }

    [Fact]
    public void AvailabilityAwareSelection_FallsBackToHighestAvailability_WhenRandomEligibilityInsufficient()
    {
        var strategy = new AvailabilityAwareClientSelectionStrategy(availabilityThreshold: 0.9);

        var request = new ClientSelectionRequest
        {
            CandidateClientIds = new[] { 10, 11, 12, 13 },
            FractionToSelect = 0.5,
            Random = RandomHelper.CreateSeededRandom(1),
            ClientAvailabilityProbabilities = new Dictionary<int, double>
            {
                [10] = -1.0,  // clamped to 0
                [11] = 0.2,
                [12] = 0.8,
                [13] = 1.5    // clamped to 1 (eligible)
            }
        };

        var selected = strategy.SelectClients(request);

        Assert.Equal("AvailabilityAware", strategy.GetStrategyName());
        Assert.Equal(2, selected.Count);
        Assert.Contains(13, selected);
    }

    [Fact]
    public void ClusteredSelection_UsesEmbeddings_WhenProvided()
    {
        var strategy = new ClusteredClientSelectionStrategy(clusterCount: 2, iterations: 2);

        var request = new ClientSelectionRequest
        {
            CandidateClientIds = new[] { 1, 2, 3, 4 },
            FractionToSelect = 0.5,
            Random = RandomHelper.CreateSeededRandom(5),
            ClientEmbeddings = new Dictionary<int, double[]>
            {
                [1] = new[] { 0.0, 0.0 },
                [2] = new[] { 0.1, 0.1 },
                [3] = new[] { 10.0, 10.0 },
                [4] = new[] { 10.1, 10.1 }
            }
        };

        var selected = strategy.SelectClients(request);

        Assert.Equal("Clustered", strategy.GetStrategyName());
        Assert.Equal(2, selected.Count);
        Assert.True(selected.SequenceEqual(selected.OrderBy(i => i)));

        Assert.Contains(selected, id => id == 1 || id == 2);
        Assert.Contains(selected, id => id == 3 || id == 4);
    }

    [Fact]
    public void WeightedRandomSelection_FallsBackToUniform_WhenNoPositiveWeights()
    {
        var strategy = new WeightedRandomClientSelectionStrategy();

        var request = new ClientSelectionRequest
        {
            CandidateClientIds = new[] { 1, 2, 3, 4 },
            FractionToSelect = 0.5,
            Random = RandomHelper.CreateSeededRandom(42),
            ClientWeights = new Dictionary<int, double>
            {
                [1] = 0.0,
                [2] = -1.0,
                [3] = 0.0,
                [4] = -5.0
            }
        };

        var selected = strategy.SelectClients(request);

        Assert.Equal("WeightedRandom", strategy.GetStrategyName());
        Assert.Equal(2, selected.Count);
        Assert.True(selected.SequenceEqual(selected.OrderBy(i => i)));
    }

    [Fact]
    public void WeightedRandomSelection_ReturnsAllCandidates_WhenDesiredExceedsCount()
    {
        var strategy = new WeightedRandomClientSelectionStrategy();

        var request = new ClientSelectionRequest
        {
            CandidateClientIds = new[] { 1, 2, 3 },
            FractionToSelect = 1.0,
            Random = RandomHelper.CreateSeededRandom(0)
        };

        var selected = strategy.SelectClients(request);

        Assert.Equal(new[] { 1, 2, 3 }, selected);
    }
}
