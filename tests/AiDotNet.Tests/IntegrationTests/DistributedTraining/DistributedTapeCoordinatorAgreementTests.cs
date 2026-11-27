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

    // A rank whose model registers no buffers used to return before the layout agreement while its peer waited in it
    // until the collective timeout; now both ranks reach the agreement and both refuse the mismatch.
    [Fact(Timeout = 120000)]
    public async Task BroadcastBuffers_RankWithNoBuffers_FailsOnEveryRankInsteadOfHanging()
    {
        await Task.Yield();
        var outcomes = OnTwoRanks((c, rank) =>
        {
            var buffers = rank == 0
                ? new[] { new AiDotNet.Tensors.LinearAlgebra.Tensor<float>(new[] { 4 }) }
                : Array.Empty<AiDotNet.Tensors.LinearAlgebra.Tensor<float>>();
            try { c.BroadcastBuffers(buffers); return "broadcast"; }
            catch (InvalidOperationException) { return "refused"; }
        });
        Assert.All(outcomes, o => Assert.Equal("refused", o));

        // Ranks that all register none agree and return without moving data.
        var none = OnTwoRanks((c, _) => { c.BroadcastBuffers(Array.Empty<AiDotNet.Tensors.LinearAlgebra.Tensor<float>>()); return true; });
        Assert.All(none, Assert.True);
    }

    // Lengths [1,3] and [2,2] have the same count (2) and total (4); broadcast onto each other, part of rank 0's second
    // buffer landed in rank 1's first. Every rank refuses the mismatch, and identical layouts still broadcast.
    [Fact(Timeout = 120000)]
    public async Task BroadcastBuffers_SameCountAndTotalButDifferentShapes_IsRefusedOnEveryRank()
    {
        await Task.Yield();
        var outcomes = OnTwoRanks((c, rank) =>
        {
            var buffers = rank == 0
                ? new[] { new AiDotNet.Tensors.LinearAlgebra.Tensor<float>(new[] { 1 }), new AiDotNet.Tensors.LinearAlgebra.Tensor<float>(new[] { 3 }) }
                : new[] { new AiDotNet.Tensors.LinearAlgebra.Tensor<float>(new[] { 2 }), new AiDotNet.Tensors.LinearAlgebra.Tensor<float>(new[] { 2 }) };
            try { c.BroadcastBuffers(buffers); return "broadcast"; }
            catch (InvalidOperationException) { return "refused"; }
        });
        Assert.All(outcomes, o => Assert.Equal("refused", o));

        var received = OnTwoRanks((c, rank) =>
        {
            var first = new AiDotNet.Tensors.LinearAlgebra.Tensor<float>(new[] { 1 });
            var second = new AiDotNet.Tensors.LinearAlgebra.Tensor<float>(new[] { 3 });
            float fill = rank == 0 ? 7f : -1f;
            first[0] = fill;
            for (int i = 0; i < 3; i++) second[i] = fill + 1 + i;
            c.BroadcastBuffers(new[] { first, second });
            return new[] { first[0], second[0], second[1], second[2] };
        });
        Assert.All(received, r => Assert.Equal(new[] { 7f, 8f, 9f, 10f }, r));
    }
}

/// <summary>The streaming options' copy keeps every setting (ConfigureStreamingTraining stores a copy).</summary>
public sealed class StreamingTrainingOptionsCopyTests
{
    [Fact]
    public void Copy_keeps_every_setting()
    {
        var source = new AiDotNet.Models.Options.StreamingTrainingOptions<float,
            AiDotNet.Tensors.LinearAlgebra.Tensor<float>, AiDotNet.Tensors.LinearAlgebra.Tensor<float>>();
        var properties = source.GetType()
            .GetProperties(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance
                | System.Reflection.BindingFlags.DeclaredOnly)
            .Where(p => p.CanRead && p.CanWrite && p.GetSetMethod() is not null)
            .ToArray();
        Assert.Contains(properties, p => p.Name == "DataParallelUpdateInCompiledPlan");
        foreach (var p in properties)
        {
            var type = Nullable.GetUnderlyingType(p.PropertyType) ?? p.PropertyType;
            if (type == typeof(bool)) p.SetValue(source, !((bool?)p.GetValue(source) ?? false));
            else if (type == typeof(int)) p.SetValue(source, 7);
            else if (type == typeof(long)) p.SetValue(source, 7L);
            else if (type == typeof(double)) p.SetValue(source, 7.5);
            else if (type == typeof(float)) p.SetValue(source, 7.5f);
            else if (!type.IsValueType && p.GetValue(source) is null) continue;   // a reference left unset copies as null
            else if (type.IsEnum)
            {
                var values = Enum.GetValues(type);
                if (values.Length > 1) p.SetValue(source, values.GetValue(values.Length - 1));
            }
        }

        var copy = new AiDotNet.Models.Options.StreamingTrainingOptions<float,
            AiDotNet.Tensors.LinearAlgebra.Tensor<float>, AiDotNet.Tensors.LinearAlgebra.Tensor<float>>(source);
        foreach (var p in properties)
            Assert.True(Equals(p.GetValue(source), p.GetValue(copy)), $"{p.Name} was not copied");
        // The validation loader is shared, not cloned.
        Assert.Same(source.ValidationLoader, copy.ValidationLoader);
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
