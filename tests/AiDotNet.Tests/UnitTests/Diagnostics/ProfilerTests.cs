using AiDotNet.Diagnostics;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diagnostics;

/// <summary>
/// Unit tests for the Profiler system.
/// Tests use a shared static Profiler, so they must run sequentially.
/// </summary>
[Collection("ProfilerTests")]
public class ProfilerTests : IDisposable
{
    private readonly string _testId = Guid.NewGuid().ToString("N")[..8];

    public ProfilerTests()
    {
        // Reset profiler state before each test
        Profiler.Disable();
        Profiler.Reset();
    }

    public void Dispose()
    {
        Profiler.Disable();
        Profiler.Reset();
    }

    [Fact]
    public void Profiler_EnableDisable_Works()
    {
        // Act
        Assert.False(Profiler.IsEnabled);
        Profiler.Enable();
        Assert.True(Profiler.IsEnabled);
        Profiler.Disable();
        Assert.False(Profiler.IsEnabled);
    }

    [Fact]
    public void ProfilerScope_RecordsTiming()
    {
        // Arrange
        Profiler.Enable();
        var opName = $"TestOperation_{_testId}";

        // Act
        using (Profiler.Scope(opName))
        {
            Thread.Sleep(50); // Sleep for 50ms
        }

        // Assert
        var stats = Profiler.GetStats(opName);
        Assert.NotNull(stats);
        Assert.Equal(1, stats.Count);
        Assert.True(stats.MeanMs >= 40, $"Expected >= 40ms but got {stats.MeanMs}ms"); // Allow some variance
    }

    [Fact]
    public void ProfilerTimer_RecordsTiming()
    {
        // Arrange
        Profiler.Enable();
        var opName = $"ManualTimer_{_testId}";

        // Act
        var timer = Profiler.Start(opName);
        Thread.Sleep(30);
        timer.Stop();

        // Assert
        var stats = Profiler.GetStats(opName);
        Assert.NotNull(stats);
        Assert.Equal(1, stats.Count);
        Assert.True(stats.MeanMs >= 20, $"Expected >= 20ms but got {stats.MeanMs}ms");
    }

    [Fact]
    public void Profiler_MultipleSamples_CalculatesStatistics()
    {
        // Arrange
        Profiler.Enable();
        var opName = $"MultiSample_{_testId}";

        // Act - Record multiple timings
        for (int i = 0; i < 10; i++)
        {
            using (Profiler.Scope(opName))
            {
                Thread.Sleep(10);
            }
        }

        // Assert
        var stats = Profiler.GetStats(opName);
        Assert.NotNull(stats);
        Assert.Equal(10, stats.Count);
        Assert.True(stats.MinMs > 0);
        Assert.True(stats.MaxMs >= stats.MinMs);
        Assert.True(stats.MeanMs >= stats.MinMs && stats.MeanMs <= stats.MaxMs);
        Assert.True(stats.P50Ms > 0);
        Assert.True(stats.P95Ms >= stats.P50Ms);
    }

    [Fact]
    public void Profiler_Reset_ClearsData()
    {
        // Arrange
        Profiler.Enable();
        using (Profiler.Scope("BeforeReset"))
        {
            Thread.Sleep(10);
        }

        // Act
        Profiler.Reset();

        // Assert
        var stats = Profiler.GetStats("BeforeReset");
        Assert.Null(stats);
    }

    [Fact]
    public void Profiler_WhenDisabled_DoesNotRecord()
    {
        // Arrange - profiler is disabled by default
        Assert.False(Profiler.IsEnabled);

        // Act
        using (Profiler.Scope("DisabledOperation"))
        {
            Thread.Sleep(10);
        }

        // Assert
        var stats = Profiler.GetStats("DisabledOperation");
        Assert.Null(stats);
    }

    [Fact]
    public void ProfileReport_GeneratesCorrectly()
    {
        // Arrange
        Profiler.Enable();
        var op1Name = $"Op1_{_testId}";
        var op2Name = $"Op2_{_testId}";

        using (Profiler.Scope(op1Name)) { Thread.Sleep(10); }
        using (Profiler.Scope(op2Name)) { Thread.Sleep(20); }
        using (Profiler.Scope(op1Name)) { Thread.Sleep(10); }

        // Act
        var report = Profiler.GetReport();

        // Assert
        Assert.NotNull(report);
        Assert.True(report.Stats.Count >= 2, $"Expected at least 2 stats but got {report.Stats.Count}");
        Assert.True(report.TotalOperations >= 3);

        var op1Stats = report.GetStats(op1Name);
        Assert.NotNull(op1Stats);
        Assert.Equal(2, op1Stats.Count);
    }

    [Fact]
    public void ProfileReport_ToJson_ProducesValidJson()
    {
        // Arrange
        Profiler.Enable();
        using (Profiler.Scope("JsonTest")) { Thread.Sleep(5); }

        // Act
        var report = Profiler.GetReport();
        var json = report.ToJson();

        // Assert
        Assert.NotNull(json);
        Assert.Contains("JsonTest", json);
        Assert.Contains("TotalOperations", json);
    }

    [Fact]
    public void ProfileReport_ToCsv_ProducesValidCsv()
    {
        // Arrange
        Profiler.Enable();
        using (Profiler.Scope("CsvTest")) { Thread.Sleep(5); }

        // Act
        var report = Profiler.GetReport();
        var csv = report.ToCsv();

        // Assert
        Assert.NotNull(csv);
        Assert.Contains("CsvTest", csv);
        Assert.Contains("Name,Count,TotalMs", csv);
    }

    [Fact]
    public void ProfileReport_ToMarkdown_ProducesValidMarkdown()
    {
        // Arrange
        Profiler.Enable();
        using (Profiler.Scope("MarkdownTest")) { Thread.Sleep(5); }

        // Act
        var report = Profiler.GetReport();
        var markdown = report.ToMarkdown();

        // Assert
        Assert.NotNull(markdown);
        Assert.Contains("# Profile Report", markdown);
        Assert.Contains("MarkdownTest", markdown);
    }

    [Fact]
    public void ProfileReport_GetHotspots_OrdersByTotalTime()
    {
        // Arrange
        Profiler.Enable();

        // Create operations with different total times
        for (int i = 0; i < 5; i++)
        {
            using (Profiler.Scope("Fast")) { } // Very fast
        }
        using (Profiler.Scope("Slow")) { Thread.Sleep(50); }

        // Act
        var report = Profiler.GetReport();
        var hotspots = report.GetHotspots(10).ToList();

        // Assert
        Assert.True(hotspots.Count >= 2);
        Assert.Equal("Slow", hotspots[0].Name); // Slow should be first (most total time)
    }

    [Fact]
    public void ProfilerStats_OpsPerSecond_CalculatesCorrectly()
    {
        // Arrange
        Profiler.Enable();

        // Create an operation that takes ~10ms
        using (Profiler.Scope("OpsPerSecTest"))
        {
            Thread.Sleep(10);
        }

        // Act
        var stats = Profiler.GetStats("OpsPerSecTest");

        // Assert
        Assert.NotNull(stats);
        // 10ms = 100 ops/sec theoretically, but CI runners have significant timing variance
        // due to shared resources, VM overhead, and thread scheduling delays
        // Use wide tolerance: > 10 ops/sec (allows up to 100ms actual sleep) and < 500 ops/sec
        Assert.True(stats.OpsPerSecond > 10 && stats.OpsPerSecond < 500,
            $"Expected ops/sec between 10 and 500, but got {stats.OpsPerSecond}");
    }

    [Fact]
    public void ProfileExtensions_ProfileAction_Works()
    {
        // Arrange
        Profiler.Enable();
        int value = 0;
        var opName = $"ActionProfile_{_testId}";

        // Act
        Action action = () => { value = 42; Thread.Sleep(10); };
        action.Profile(opName);

        // Assert
        Assert.Equal(42, value);
        var stats = Profiler.GetStats(opName);
        Assert.NotNull(stats);
        Assert.Equal(1, stats.Count);
    }

    [Fact]
    public void ProfileExtensions_ProfileFunc_Works()
    {
        // Arrange
        Profiler.Enable();

        // Act
        Func<int> func = () => { Thread.Sleep(10); return 42; };
        int result = func.Profile("FuncProfile");

        // Assert
        Assert.Equal(42, result);
        var stats = Profiler.GetStats("FuncProfile");
        Assert.NotNull(stats);
    }

    [Fact]
    public async Task ProfileExtensions_ProfileAsync_Works()
    {
        // Arrange
        Profiler.Enable();

        // Act
        Func<Task> asyncFunc = async () => await Task.Delay(20);
        await asyncFunc.ProfileAsync("AsyncProfile");

        // Assert
        var stats = Profiler.GetStats("AsyncProfile");
        Assert.NotNull(stats);
        Assert.True(stats.MeanMs >= 15);
    }

    [Fact]
    public void ProfileReport_CompareTo_DetectsRegression()
    {
        // Create baseline report
        Profiler.Enable();
        using (Profiler.Scope("CompareOp")) { Thread.Sleep(10); }
        var baseline = Profiler.GetReport();

        // Reset and create current report with slower operation
        Profiler.Reset();
        using (Profiler.Scope("CompareOp")) { Thread.Sleep(50); }
        var current = Profiler.GetReport();

        // Act
        var comparison = current.CompareTo(baseline, thresholdPercent: 50);

        // Assert - should detect regression (more than 50% slower)
        Assert.True(comparison.RegressionCount > 0);
    }
}

/// <summary>
/// Unit tests for MemoryTracker.
/// </summary>
public class MemoryTrackerTests : IDisposable
{
    public MemoryTrackerTests()
    {
        MemoryTracker.Disable();
        MemoryTracker.Reset();
    }

    public void Dispose()
    {
        MemoryTracker.Disable();
        MemoryTracker.Reset();
    }

    [Fact]
    public void MemoryTracker_Snapshot_ReturnsValidData()
    {
        // Act
        var snapshot = MemoryTracker.Snapshot("Test");

        // Assert
        Assert.Equal("Test", snapshot.Label);
        Assert.True(snapshot.TotalMemory > 0);
        Assert.True(snapshot.WorkingSet > 0);
        Assert.True(snapshot.Timestamp <= DateTime.UtcNow);
    }

    [Fact]
    public void MemorySnapshot_CompareTo_CalculatesDiff()
    {
        // Arrange
        var before = MemoryTracker.Snapshot("before");

        // Allocate some memory
        var data = new byte[1024 * 1024]; // 1 MB
        GC.KeepAlive(data);

        var after = MemoryTracker.Snapshot("after");

        // Act
        var diff = after.CompareTo(before);

        // Assert
        Assert.NotNull(diff);
        Assert.True(diff.Duration.TotalMilliseconds >= 0);
        // Note: Memory diff might be negative due to GC, so we just check it calculated
    }

    [Fact]
    public void MemoryTracker_GetPressureLevel_ReturnsValidLevel()
    {
        // Act
        var level = MemoryTracker.GetPressureLevel();

        // Assert
        Assert.True(Enum.IsDefined(typeof(MemoryPressureLevel), level));
    }

    [Fact]
    public void MemoryTracker_EstimateTensorMemory_CalculatesCorrectly()
    {
        // Arrange
        int[] shape = { 2, 3, 4, 5 }; // 2*3*4*5 = 120 elements
        int elementSize = 4; // float32

        // Act
        long estimate = MemoryTracker.EstimateTensorMemory(shape, elementSize);

        // Assert
        Assert.Equal(120 * 4, estimate); // 480 bytes
    }

    [Fact]
    public void MemoryTracker_EstimateKVCacheMemory_CalculatesCorrectly()
    {
        // Arrange - Small model config
        int numLayers = 12;
        int numHeads = 12;
        int headDim = 64;
        int maxSeqLen = 1024;
        int batchSize = 1;
        int bytesPerElement = 4;

        // Expected: 12 layers * (1 * 12 * 1024 * 64 * 4 * 2) for K and V
        long expected = (long)numLayers * batchSize * numHeads * maxSeqLen * headDim * bytesPerElement * 2;

        // Act
        long estimate = MemoryTracker.EstimateKVCacheMemory(
            numLayers, numHeads, headDim, maxSeqLen, batchSize, bytesPerElement);

        // Assert
        Assert.Equal(expected, estimate);
    }

    [Fact]
    public void MemoryTracker_History_RecordsWhenEnabled()
    {
        // Arrange
        MemoryTracker.Enable();

        // Act
        MemoryTracker.Snapshot("snap1");
        MemoryTracker.Snapshot("snap2");

        var history = MemoryTracker.GetHistory();

        // Assert
        Assert.Equal(2, history.Count);
        Assert.Equal("snap1", history[0].Label);
        Assert.Equal("snap2", history[1].Label);
    }

    [Fact]
    public void MemoryScope_TracksMemory()
    {
        // Arrange
        MemoryTracker.Enable();

        // Act
        using (var scope = MemoryTracker.TrackScope("TestScope"))
        {
            // Allocate some memory
            var data = new int[10000];
            GC.KeepAlive(data);
        }

        var history = MemoryTracker.GetHistory();

        // Assert - should have before and after snapshots
        Assert.True(history.Count >= 2);
    }
}
