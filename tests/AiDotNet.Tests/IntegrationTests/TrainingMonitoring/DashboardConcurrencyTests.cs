using System.Collections.Concurrent;
using System.Reflection;
using AiDotNet.TrainingMonitoring.Dashboard;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.TrainingMonitoring;

public class DashboardConcurrencyTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), $"DashboardConcurrency_{Guid.NewGuid():N}");

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void SnapshotWaitsForWriterAndReturnsAnIndependentList(bool html, bool histogram)
    {
        using var dashboard = Create(html);
        dashboard.LogScalar("loss", 0, 1);
        dashboard.LogHistogram("weights", 0, new[] { 1.0, 2.0 });
        object series = histogram
            ? Field<ConcurrentDictionary<string, List<HistogramDataPoint>>>(dashboard, "_histograms")["weights"]
            : Field<ConcurrentDictionary<string, List<ScalarDataPoint>>>(dashboard, "_scalars")["loss"];
        using var entered = new ManualResetEventSlim();
        Task reader;
        Monitor.Enter(series);
        try
        {
            reader = Task.Run(() =>
            {
                entered.Set();
                if (histogram) Assert.Single(dashboard.GetHistogramData()["weights"]);
                else Assert.Single(dashboard.GetScalarData()["loss"]);
            });
            Assert.True(entered.Wait(TimeSpan.FromSeconds(5)), "Reader did not start.");
            Assert.False(reader.Wait(TimeSpan.FromMilliseconds(100)), "Snapshot bypassed the writer's lock.");
        }
        finally { Monitor.Exit(series); }
        Assert.True(reader.Wait(TimeSpan.FromSeconds(5)), "Snapshot did not finish after writer released the lock.");
        if (histogram)
        {
            var copy = dashboard.GetHistogramData()["weights"];
            copy.Clear();
            Assert.Single(dashboard.GetHistogramData()["weights"]);
        }
        else
        {
            var copy = dashboard.GetScalarData()["loss"];
            copy.Clear();
            Assert.Single(dashboard.GetScalarData()["loss"]);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task LoggingClearAndReportsCanOverlap(bool html)
    {
        using var dashboard = Create(html);
        using var start = new ManualResetEventSlim();
        var writer = Task.Run(() =>
        {
            start.Wait();
            for (int step = 0; step < 2000; step++)
            {
                dashboard.LogScalar("loss", step, step);
                dashboard.LogHistogram("weights", step, new[] { (double)step, step + 1.0 });
                if (step % 31 == 0) dashboard.Clear();
            }
        });
        var reader = Task.Run(() =>
        {
            start.Wait();
            for (int i = 0; i < 10; i++)
            {
                foreach (var series in dashboard.GetScalarData().Values)
                    Assert.All(series, point => Assert.NotNull(point));
                foreach (var series in dashboard.GetHistogramData().Values)
                    Assert.All(series, point => Assert.NotNull(point));
                dashboard.GenerateReport();
                dashboard.Flush();
            }
        });
        start.Set();
        await Task.WhenAll(writer, reader);
        dashboard.LogScalar("final", 2001, 42);
        Assert.Equal(42, Assert.Single(dashboard.GetScalarData()["final"]).Value);
    }

    [Fact]
    public void RestartChangesGenerationAndStoppedCallbacksCannotRender()
    {
        using var dashboard = (ConsoleDashboard)Create(false);
        dashboard.Start();
        var previousGeneration = Field<object>(dashboard, "_renderGeneration");
        dashboard.Stop();
        Assert.False(dashboard.IsRunning);
        dashboard.Start();
        Assert.True(dashboard.IsRunning);
        Assert.NotSame(previousGeneration, Field<object>(dashboard, "_renderGeneration"));
        dashboard.LogScalar("loss", 0, 1);
        var series = Field<ConcurrentDictionary<string, List<ScalarDataPoint>>>(dashboard, "_scalars")["loss"];
        // Drain the immediate new-timer render before holding its series lock.
        dashboard.Stop();
        Monitor.Enter(series);
        try
        {
            var callback = typeof(ConsoleDashboard).GetMethod("RenderTick", BindingFlags.Instance | BindingFlags.NonPublic)
                ?? throw new InvalidOperationException("Missing timer callback.");
            var stale = Task.Run(() => callback.Invoke(dashboard, new[] { previousGeneration }));
            Assert.True(stale.Wait(TimeSpan.FromSeconds(5)), "Stopped callback attempted to read live series.");
        }
        finally { Monitor.Exit(series); }
        dashboard.Dispose();
        dashboard.Dispose();
        Assert.False(dashboard.IsRunning);
        Assert.Throws<ObjectDisposedException>(() => dashboard.Start());
    }

    [Fact]
    public void StopWaitsForAnInFlightTimerRender()
    {
        using var dashboard = (ConsoleDashboard)Create(false);
        dashboard.LogScalar("loss", 0, 1);
        var series = Field<ConcurrentDictionary<string, List<ScalarDataPoint>>>(dashboard, "_scalars")["loss"];
        var renderLock = Field<object>(dashboard, "_renderLock");
        Task stop;
        Monitor.Enter(series);
        try
        {
            dashboard.Start();
            Assert.True(SpinWait.SpinUntil(() =>
            {
                if (!Monitor.TryEnter(renderLock)) return true;
                Monitor.Exit(renderLock);
                return false;
            }, TimeSpan.FromSeconds(5)), "Timer did not enter rendering.");
            stop = Task.Run(() => dashboard.Stop());
            Assert.False(stop.Wait(TimeSpan.FromMilliseconds(100)), "Stop returned with a render still in flight.");
        }
        finally { Monitor.Exit(series); }
        Assert.True(stop.Wait(TimeSpan.FromSeconds(5)), "Stop did not complete after the render was released.");
        Assert.False(dashboard.IsRunning);
    }

    [Fact]
    public void InvalidTimerIntervalDoesNotPublishRunningState()
    {
        using var dashboard = (ConsoleDashboard)Create(false);
        dashboard.RefreshIntervalMs = -2;
        Assert.Throws<ArgumentOutOfRangeException>(() => dashboard.Start());
        Assert.False(dashboard.IsRunning);
        dashboard.RefreshIntervalMs = 1000;
        dashboard.Start();
        Assert.True(dashboard.IsRunning);
    }

    private ITrainingDashboard Create(bool html) => html
        ? new HtmlDashboard(_directory, "concurrency")
        : new ConsoleDashboard(_directory, "concurrency")
        {
            UseColors = false, ClearOnRender = false, ChartWidth = 4, ChartHeight = 1,
            RefreshIntervalMs = Timeout.Infinite
        };

    private static T Field<T>(object instance, string name) =>
        instance.GetType().GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)?.GetValue(instance) is T value
            ? value : throw new InvalidOperationException($"Missing dashboard field {name}.");

    public void Dispose()
    {
        if (Directory.Exists(_directory)) Directory.Delete(_directory, recursive: true);
    }
}
