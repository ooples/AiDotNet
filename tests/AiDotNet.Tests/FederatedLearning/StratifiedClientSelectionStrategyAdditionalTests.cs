namespace AiDotNet.Tests.FederatedLearning;

using AiDotNet.FederatedLearning.Selection;
using AiDotNet.Models;
using AiDotNet.Tensors.Helpers;
using Xunit;

public class StratifiedClientSelectionStrategyAdditionalTests
{
    [Fact]
    public void SelectClients_Throws_WhenRequestNull()
    {
        var strategy = new StratifiedClientSelectionStrategy();

        Assert.Throws<ArgumentNullException>(() => strategy.SelectClients(request: null!));
    }

    [Fact]
    public void SelectClients_FallsBackToShuffle_WhenGroupKeysNull()
    {
        var strategy = new StratifiedClientSelectionStrategy();

        var request = new ClientSelectionRequest
        {
            CandidateClientIds = new[] { 1, 2, 3, 4 },
            FractionToSelect = 0.5,
            Random = RandomHelper.CreateSeededRandom(123),
            ClientGroupKeys = null
        };

        var selected = strategy.SelectClients(request);

        Assert.Equal(2, selected.Count);
        Assert.All(selected, id => Assert.Contains(id, request.CandidateClientIds));
        Assert.True(selected.SequenceEqual(selected.OrderBy(i => i)));
    }

    [Fact]
    public void SelectClients_FallsBackToShuffle_WhenGroupKeysEmpty()
    {
        var strategy = new StratifiedClientSelectionStrategy();

        var request = new ClientSelectionRequest
        {
            CandidateClientIds = new[] { 1, 2, 3, 4 },
            FractionToSelect = 0.5,
            Random = RandomHelper.CreateSeededRandom(123),
            ClientGroupKeys = new Dictionary<int, string>()
        };

        var selected = strategy.SelectClients(request);

        Assert.Equal(2, selected.Count);
        Assert.All(selected, id => Assert.Contains(id, request.CandidateClientIds));
        Assert.True(selected.SequenceEqual(selected.OrderBy(i => i)));
    }

    [Fact]
    public void SelectClients_FallsBackToShuffle_WhenOnlyOneGroup()
    {
        var strategy = new StratifiedClientSelectionStrategy();

        var request = new ClientSelectionRequest
        {
            CandidateClientIds = new[] { 1, 2, 3, 4 },
            FractionToSelect = 0.5,
            Random = RandomHelper.CreateSeededRandom(5),
            ClientGroupKeys = new Dictionary<int, string>
            {
                [1] = "one",
                [2] = "one",
                [3] = "one",
                [4] = "one"
            }
        };

        var selected = strategy.SelectClients(request);

        Assert.Equal(2, selected.Count);
        Assert.All(selected, id => Assert.Contains(id, request.CandidateClientIds));
        Assert.True(selected.SequenceEqual(selected.OrderBy(i => i)));
    }

    [Fact]
    public void SelectClients_AssignsDefaultGroup_WhenGroupKeyMissingOrBlank()
    {
        var strategy = new StratifiedClientSelectionStrategy();

        var request = new ClientSelectionRequest
        {
            CandidateClientIds = new[] { 1, 2, 3, 4 },
            FractionToSelect = 0.5,
            Random = RandomHelper.CreateSeededRandom(42),
            ClientGroupKeys = new Dictionary<int, string>
            {
                [1] = "A",
                [2] = "A",
                [4] = "   "
            }
        };

        var selected = strategy.SelectClients(request);

        Assert.Equal(2, selected.Count);
        Assert.Contains(selected, id => id == 3 || id == 4);
        Assert.True(selected.SequenceEqual(selected.OrderBy(i => i)));
    }

    [Fact]
    public void SelectClients_IncreasesGroupAllocations_WhenAllocatedLessThanDesired()
    {
        var strategy = new StratifiedClientSelectionStrategy();

        var candidates = Enumerable.Range(1, 10).ToArray();
        var groupKeys = new Dictionary<int, string>();
        foreach (var id in candidates)
        {
            if (id <= 2)
            {
                groupKeys[id] = "g1";
            }
            else if (id <= 4)
            {
                groupKeys[id] = "g2";
            }
            else
            {
                groupKeys[id] = "g3";
            }
        }

        var request = new ClientSelectionRequest
        {
            CandidateClientIds = candidates,
            FractionToSelect = 0.7, // desired=7
            Random = RandomHelper.CreateSeededRandom(7),
            ClientGroupKeys = groupKeys
        };

        var selected = strategy.SelectClients(request);

        Assert.Equal(7, selected.Count);
        Assert.Contains(1, selected);
        Assert.Contains(2, selected);
        Assert.Contains(selected, id => id == 3 || id == 4);
        Assert.Contains(selected, id => id >= 5 && id <= 10);
        Assert.True(selected.SequenceEqual(selected.OrderBy(i => i)));
    }
}

