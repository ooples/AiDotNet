using AttributionSubject;
using Xunit;

namespace PrototypeTests;

public sealed class LateTests : IAsyncLifetime
{
    private readonly TaskCompletionSource release = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private Task? background;

    public Task InitializeAsync() => Task.CompletedTask;

    [Fact, Trait("Scenario", "Late")]
    public void LateBackground()
    {
        // Prime the deduplication cache; a repeated late hit must still be detected.
        Assert.Equal(61, CodePaths.Late());
        background = Task.Run(async () =>
        {
            await release.Task;
            Assert.Equal(61, CodePaths.Late());
        });
    }

    public async Task DisposeAsync()
    {
        // xUnit runs After before class disposal; execution context still contains
        // the closed scope in the detached task. This must poison attribution.
        release.SetResult();
        if (background is not null) await background.WaitAsync(TimeSpan.FromSeconds(15));
    }
}

public sealed class FailingTests
{
    [Fact, Trait("Scenario", "Failure")]
    public void FailsAfterCoverage()
    {
        Assert.Equal(12, Operations.Left(1));
        throw new InvalidOperationException("Deliberate test failure: attribution alone is not validation success.");
    }
}

public sealed class DetachedTaskTests
{
    [Fact, Trait("Scenario", "DetachedTask")]
    public void NeverHitsCoveredCode()
    {
        var never = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        _ = Task.Run(async () => await never.Task);
    }
}
