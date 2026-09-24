using AttributionSubject;
using Xunit;

namespace PrototypeTests;

public sealed class LateReleaseFixture : IAsyncLifetime
{
    private readonly TaskCompletionSource release = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private Task? background;

    public Task InitializeAsync() => Task.CompletedTask;

    public void Start()
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
        // Shared fixture cleanup occurs after the individual test case has closed.
        release.SetResult();
        if (background is not null) await background.WaitAsync(TimeSpan.FromSeconds(15));
    }
}

public sealed class LateTests(LateReleaseFixture fixture) : IClassFixture<LateReleaseFixture>
{
    [Fact, Trait("Scenario", "Late")]
    public void LateBackground() => fixture.Start();
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

public sealed class UntrackedBoundaryTests
{
    [Fact, Trait("Scenario", "AfterPublication")]
    public void ExitCallbackHitsAfterReport()
    {
        // The framework initialized Tracker before this test, so its exit
        // publication handler runs before the callback registered here.
        Assert.Equal(61, CodePaths.Late());
        AppDomain.CurrentDomain.ProcessExit += (_, _) => CodePaths.Late();
    }

    [Fact, Trait("Scenario", "UntrackedTimer")]
    public void TimerNeverFires()
    {
        using var timer = new Timer(_ => throw new InvalidOperationException("Must never fire."),
            null, Timeout.Infinite, Timeout.Infinite);
    }

    [Fact, Trait("Scenario", "UntrackedProcess")]
    public async Task ProcessNeverProducesCoverage()
    {
        var start = new System.Diagnostics.ProcessStartInfo("dotnet") { UseShellExecute = false, CreateNoWindow = true };
        start.ArgumentList.Add(Environment.GetEnvironmentVariable("ATTRIBUTION_WORKER_DLL")
            ?? throw new InvalidOperationException("Missing worker fixture."));
        start.ArgumentList.Add("missing");
        using var child = System.Diagnostics.Process.Start(start)
            ?? throw new InvalidOperationException("Child did not start.");
        using var deadline = new CancellationTokenSource(TimeSpan.FromSeconds(15));
        try { await child.WaitForExitAsync(deadline.Token); }
        catch (OperationCanceledException)
        {
            child.Kill(entireProcessTree: true);
            await child.WaitForExitAsync();
            throw;
        }
        Assert.Equal(0, child.ExitCode);
    }
}
