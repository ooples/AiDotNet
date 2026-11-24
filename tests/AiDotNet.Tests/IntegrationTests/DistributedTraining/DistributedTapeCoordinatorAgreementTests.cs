using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.DistributedTraining;
using AiDotNet.Enums;
using Xunit;

namespace AiDotNetTests.IntegrationTests.DistributedTraining;

/// <summary>
/// The ranks' agreement checks the compiled data-parallel step runs before an exchange: a layout count compared exactly
/// (as 7-bit chunks, not as one element of the model's type), and one flag every rank must hold.
/// </summary>
public sealed class DistributedTapeCoordinatorAgreementTests
{
    private static T[] OnTwoRanks<T>(Func<DistributedTapeCoordinator<float>, int, T> body)
    {
        string session = "agree_" + Guid.NewGuid().ToString("N");
        var backends = Enumerable.Range(0, 2).Select(r => new InMemoryCommunicationBackend<float>(r, 2, session)).ToArray();
        try
        {
            var tasks = Enumerable.Range(0, 2).Select(rank => Task.Run(() =>
            {
                backends[rank].Initialize();
                var coordinator = new DistributedTapeCoordinator<float>(backends[rank], DistributedStrategy.DDP);
                return body(coordinator, rank);
            })).ToArray();
            Assert.True(Task.WaitAll(tasks, TimeSpan.FromSeconds(60)), "a rank did not finish the collective");
            return tasks.Select(t => t.Result).ToArray();
        }
        finally
        {
            foreach (var backend in backends)
            {
                try { if (backend.IsInitialized) backend.Shutdown(); } catch (InvalidOperationException) { }
            }
        }
    }

    [Fact(Timeout = 120000)]
    public async Task RanksAgree_TellsApartCountsAFloatCannot()
    {
        await Task.Yield();
        // 2^24 and 2^24 + 1 are the same float: compared as one float element the ranks "agreed".
        var differ = OnTwoRanks((c, rank) => c.RanksAgree(rank == 0 ? 16_777_216L : 16_777_217L, 5));
        Assert.All(differ, Assert.False);

        var same = OnTwoRanks((c, _) => c.RanksAgree(16_777_217L, long.MaxValue, -3));
        Assert.All(same, Assert.True);
    }

    [Fact(Timeout = 120000)]
    public async Task AllRanks_IsFalseOnEveryRankWhenOneRankIsFalse()
    {
        await Task.Yield();
        Assert.All(OnTwoRanks((c, rank) => c.AllRanks(rank == 0)), Assert.False);
        Assert.All(OnTwoRanks((c, _) => c.AllRanks(true)), Assert.True);
    }
}

/// <summary>The dropout base seed follows the configured seed (a fixed seed reproduces the masks) and stays in range.</summary>
public sealed class StochasticBaseSeedTests
{
    private static int Draw(int? seed)
        => AiDotNet.AiModelBuilder<float, AiDotNet.Tensors.LinearAlgebra.Tensor<float>, AiDotNet.Tensors.LinearAlgebra.Tensor<float>>
            .DrawStochasticBaseSeed(seed);

    [Fact]
    public void A_configured_seed_gives_the_same_base_seed_and_different_seeds_differ()
    {
        foreach (int seed in new[] { 0, 1, 42, -7, int.MaxValue, int.MinValue })
        {
            int first = Draw(seed);
            Assert.Equal(first, Draw(seed));
            Assert.InRange(first, 1, int.MaxValue - 1);
        }
        Assert.NotEqual(Draw(1), Draw(2));
        Assert.NotEqual(42, Draw(42));   // apart from the data-order seed itself
        Assert.InRange(Draw(null), 1, int.MaxValue - 1);
    }
}
