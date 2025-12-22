namespace AiDotNet.Tests.FederatedLearning;

using AiDotNet.Tensors.Helpers;
using Xunit;

public class ClientSelectionStrategyBaseTests
{
    [Fact]
    public void GetDesiredClientCount_Throws_WhenCandidatesNull()
    {
        Assert.Throws<ArgumentNullException>(() =>
            ClientSelectionStrategyBaseAccessor.CallGetDesiredClientCount(candidates: null!, fraction: 0.5));
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(-0.1)]
    [InlineData(1.1)]
    public void GetDesiredClientCount_Throws_WhenFractionOutOfRange(double fraction)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            ClientSelectionStrategyBaseAccessor.CallGetDesiredClientCount(candidates: new[] { 1, 2, 3 }, fraction: fraction));
    }

    [Fact]
    public void GetDesiredClientCount_ReturnsCeilingWithMinimumOne()
    {
        var candidates = Enumerable.Range(0, 10).ToArray();

        Assert.Equal(1, ClientSelectionStrategyBaseAccessor.CallGetDesiredClientCount(candidates, fraction: 0.01));
        Assert.Equal(5, ClientSelectionStrategyBaseAccessor.CallGetDesiredClientCount(candidates, fraction: 0.5));
        Assert.Equal(6, ClientSelectionStrategyBaseAccessor.CallGetDesiredClientCount(candidates, fraction: 0.51));
        Assert.Equal(10, ClientSelectionStrategyBaseAccessor.CallGetDesiredClientCount(candidates, fraction: 1.0));
    }

    [Fact]
    public void ShuffleAndTake_Throws_WhenItemsNull()
    {
        Assert.Throws<ArgumentNullException>(() =>
            ClientSelectionStrategyBaseAccessor.CallShuffleAndTake(items: null!, count: 1, random: RandomHelper.CreateSeededRandom(1)));
    }

    [Fact]
    public void ShuffleAndTake_Throws_WhenRandomNull()
    {
        Assert.Throws<ArgumentNullException>(() =>
            ClientSelectionStrategyBaseAccessor.CallShuffleAndTake(items: new[] { 1 }, count: 1, random: null!));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void ShuffleAndTake_Throws_WhenCountNotPositive(int count)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            ClientSelectionStrategyBaseAccessor.CallShuffleAndTake(items: new[] { 1, 2 }, count: count, random: RandomHelper.CreateSeededRandom(1)));
    }

    [Fact]
    public void ShuffleAndTake_ReturnsSortedItems_WhenCountAtLeastItemCount()
    {
        var items = new[] { 3, 1, 2 };

        var selected = ClientSelectionStrategyBaseAccessor.CallShuffleAndTake(
            items: items,
            count: 3,
            random: RandomHelper.CreateSeededRandom(1));

        Assert.Equal(new[] { 1, 2, 3 }, selected);
    }

    [Fact]
    public void ShuffleAndTake_ReturnsSortedSubset_WhenCountLessThanItemCount()
    {
        var items = new[] { 1, 2, 3, 4, 5 };

        var selected = ClientSelectionStrategyBaseAccessor.CallShuffleAndTake(
            items: items,
            count: 2,
            random: RandomHelper.CreateSeededRandom(123));

        Assert.Equal(2, selected.Count);
        Assert.True(selected.SequenceEqual(selected.OrderBy(i => i)));
        Assert.All(selected, id => Assert.Contains(id, items));
        Assert.Equal(selected.Count, selected.Distinct().Count());
    }
}

