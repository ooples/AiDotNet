using System;
using System.Linq;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests;

/// <summary>
/// The parameter-count sweep splits models by a stable hash of their name, and the shard selector computes the
/// same hash to find which window a changed model lands in. These keep the two equal and the split usable.
/// </summary>
public class StableModelShardingTests
{
    [Fact]
    public void ShardOfMatchesTheSelectorsKnownAnswer()
    {
        // FNV-1a("a") = 0xE40C292C; tools/TestImpact/Test-AuxiliaryInventory.ps1 asserts the same shard.
        Assert.Equal(4, ParameterCountContractTests.ShardOf("a"));
        // A closed model type hashes as its open definition, the name the selector reads from source.
        Assert.Equal(ParameterCountContractTests.ShardOf("System.Collections.Generic.List`1"),
            ParameterCountContractTests.ShardOf(typeof(System.Collections.Generic.List<double>)));

    }

    [Fact]
    public void EveryShardGetsAShareOfTheModels()
    {
        var models = ParameterCountContractTests.GetConstructableModelTypes().ToList();
        var sizes = Enumerable.Range(0, ParameterCountContractTests.ShardCount)
            .Select(shard => models.Count(t => ParameterCountContractTests.ShardOf(t) == shard))
            .ToArray();
        double mean = models.Count / (double)ParameterCountContractTests.ShardCount;
        // A hash split is not exact; each shard must stay within half to one and a half times the mean,
        // which the 30-minute per-shard budget absorbs.
        Assert.True(sizes.All(n => n >= mean * 0.5 && n <= mean * 1.5),
            $"shard sizes [{string.Join(", ", sizes)}] for {models.Count} models stray beyond 0.5-1.5x the mean {mean:F1}");
    }
}