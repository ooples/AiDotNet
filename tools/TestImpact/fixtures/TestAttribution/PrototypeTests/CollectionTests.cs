using AttributionSubject;
using Xunit;

[assembly: CollectionBehavior(MaxParallelThreads = 4)]

namespace PrototypeTests;

public sealed class SharedFixture : IDisposable
{
    public SharedFixture() => Assert.Equal(31, Operations.Setup());
    public void Dispose() => Assert.Equal(47, Operations.Cleanup());
}

public sealed class MethodTests : IClassFixture<SharedFixture>
{
    public MethodTests(SharedFixture fixture) => ArgumentNullException.ThrowIfNull(fixture);

    [Theory, Trait("Scenario", "Positive")]
    [InlineData(1)]
    [InlineData(2)]
    public async Task First(int value) => Assert.Equal(value + 11, await CodePaths.AsyncLeft(value));

    [Fact, Trait("Scenario", "Positive")]
    public async Task Second()
    {
        await Task.Yield();
        Assert.Equal(24, await Task.Run(() => Operations.Right(1)));
    }

    [Fact, Trait("Scenario", "Positive")]
    public void PotentialCaller() => Assert.Equal(0, CodePaths.UntakenBranch(false));

    [Fact, Trait("Scenario", "Positive")]
    public async Task SuppressedContext()
    {
        Task<int> work;
        using (ExecutionContext.SuppressFlow()) work = Task.Run(CodePaths.Unowned);
        Assert.Equal(59, await work);
    }
}

internal static class ParallelRendezvous
{
    private static readonly TaskCompletionSource Left = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private static readonly TaskCompletionSource Right = new(TaskCreationOptions.RunContinuationsAsynchronously);

    public static async Task Meet(bool left)
    {
        (left ? Left : Right).SetResult();
        await (left ? Right : Left).Task.WaitAsync(TimeSpan.FromSeconds(15));
    }
}

public sealed class ParallelLeftTests
{
    [Fact, Trait("Scenario", "Positive")]
    public async Task Left()
    {
        await ParallelRendezvous.Meet(true);
        Assert.Equal(12, Operations.Left(1));
    }
}

public sealed class ParallelRightTests
{
    [Fact, Trait("Scenario", "Positive")]
    public async Task Right()
    {
        await ParallelRendezvous.Meet(false);
        Assert.Equal(24, Operations.Right(1));
    }
}
