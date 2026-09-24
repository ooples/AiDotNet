using AttributionSubject;
using Xunit;

namespace PrototypeTests;

public abstract class InheritedBaseTests
{
    protected abstract int Expected { get; }
    protected abstract int Compute();

    [Fact, Trait("Scenario", "Positive")]
    public void Inherited() => Assert.Equal(Expected, Compute());
}

public sealed class InheritedLeftTests : InheritedBaseTests
{
    protected override int Expected => 12;
    protected override int Compute() => Operations.Left(1);
}

public sealed class InheritedRightTests : InheritedBaseTests
{
    protected override int Expected => 24;
    protected override int Compute() => Operations.Right(1);
}

public sealed class CleanupTests : IAsyncLifetime
{
    private readonly TaskCompletionSource release = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private Task? work;
    public Task InitializeAsync() => Task.CompletedTask;

    [Fact, Trait("Scenario", "Positive")]
    public void CleanupJoinsBackground()
    {
        work = Task.Run(async () =>
        {
            await release.Task;
            Assert.Equal(59, CodePaths.Unowned());
        });
    }

    public async Task DisposeAsync()
    {
        release.SetResult();
        if (work is not null) await work.WaitAsync(TimeSpan.FromSeconds(15));
    }
}
