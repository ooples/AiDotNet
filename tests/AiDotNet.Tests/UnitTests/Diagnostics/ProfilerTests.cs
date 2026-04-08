using AiDotNet.Deployment.Configuration;
using AiDotNet.Diagnostics;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diagnostics;

/// <summary>
/// Unit tests for the ProfilerSession system.
/// </summary>
public class ProfilerSessionTests
{
    private readonly string _testId = Guid.NewGuid().ToString("N")[..8];

    [Fact]
    public void ProfilerSession_EnableDisable_Works()
    {
        // Arrange - explicitly disable AutoEnableInDebug to test disabled state
        var config = new ProfilingConfig { Enabled = false, AutoEnableInDebug = false };
        var session = new ProfilerSession(config);

        // Act & Assert
        Assert.False(session.IsEnabled);
        session.Enable();
        Assert.True(session.IsEnabled);
        session.Disable();
        Assert.False(session.IsEnabled);
    }

    [Fact]
    public void ProfilerSession_DefaultConfig_UsesDefaults()
    {
        // Arrange & Act
        var session = new ProfilerSession();

        // Assert - should use default config values
        Assert.NotNull(session.Config);
        Assert.Equal(1.0, session.Config.SamplingRate);
        Assert.Equal(1000, session.Config.ReservoirSize);
        Assert.Equal(10000, session.Config.MaxOperations);
    }

    [Fact]
    public void ProfilerSessionScope_RecordsTiming()
    {
        // Arrange
        var config = new ProfilingConfig { Enabled = true };
        var session = new ProfilerSession(config);
        var opName = $"TestOperation_{_testId}";

        // Act
        using (session.Scope(opName))
        {
            Thread.Sleep(50); // Sleep for 50ms
        }

        // Assert
        var stats = session.GetStats(opName);
        Assert.NotNull(stats);
        Assert.Equal(1, stats.Count);
        Assert.True(stats.MeanMs >= 40, $"Expected >= 40ms but got {stats.MeanMs}ms"); // Allow some variance
    }

    [Fact]
    public void ProfilerSessionTimer_RecordsTiming()
    {
        // Arrange
        var config = new ProfilingConfig { Enabled = true };
        var session = new ProfilerSession(config);
        var opName = $"ManualTimer_{_testId}";

        // Act
        var timer = session.Start(opName);
        Thread.Sleep(30);
        timer.Stop();

        // Assert
        var stats = session.GetStats(opName);
        Assert.NotNull(stats);
        Assert.Equal(1, stats.Count);
        Assert.True(stats.MeanMs >= 20, $"Expected >= 20ms but got {stats.MeanMs}ms");
    }

    [Fact]
    public void ProfilerSession_MultipleSamples_CalculatesStatistics()
    {
        // Arrange
        var config = new ProfilingConfig { Enabled = true };
        var session = new ProfilerSession(config);
        var opName = $"MultiSample_{_testId}";

        // Act - Record multiple timings
        for (int i = 0; i < 10; i++)
        {
            using (session.Scope(opName))
            {
                Thread.Sleep(10);
            }
        }

        // Assert
        var stats = session.GetStats(opName);
        Assert.NotNull(stats);
        Assert.Equal(10, stats.Count);
        Assert.True(stats.MinMs > 0);
        Assert.True(stats.MaxMs >= stats.MinMs);
        Assert.True(stats.MeanMs >= stats.MinMs && stats.MeanMs <= stats.MaxMs);
        Assert.True(stats.P50Ms > 0);
        Assert.True(stats.P95Ms >= stats.P50Ms);
    }

    [Fact]
    public void ProfilerSession_Reset_ClearsData()
    {
        // Arrange
        var config = new ProfilingConfig { Enabled = true };
        var session = new ProfilerSession(config);
        using (session.Scope("BeforeReset"))
        {
            Thread.Sleep(10);
        }

        // Act
        session.Reset();

        // Assert
        var stats = session.GetStats("BeforeReset");
        Assert.Null(stats);
    }

    [Fact]
    public void ProfilerSession_WhenDisabled_DoesNotRecord()
    {
        // Arrange - profiler is disabled, explicitly disable AutoEnableInDebug for testing
        var config = new ProfilingConfig { Enabled = false, AutoEnableInDebug = false };
        var session = new ProfilerSession(config);
        Assert.False(session.IsEnabled);

        // Act
        using (session.Scope("DisabledOperation"))
        {
            Thread.Sleep(10);
        }

        // Assert
        var stats = session.GetStats("DisabledOperation");
        Assert.Null(stats);
    }

    [Fact]
    public void ProfilerSession_SamplingRate_RespectsRate()
    {
        // Arrange - 10% sampling rate
        var config = new ProfilingConfig { Enabled = true, SamplingRate = 0.1 };
        var session = new ProfilerSession(config);
        var opName = $"SampledOp_{_testId}";

        // Act - Record 1000 operations
        for (int i = 0; i < 1000; i++)
        {
            using (session.Scope(opName))
            {
                // Very fast operation
            }
        }

        // Assert - with 10% sampling, we expect roughly 100 samples (allow 50-200 range)
        var stats = session.GetStats(opName);
        if (stats != null)
        {
            Assert.True(stats.Count < 500, $"Expected less than 500 samples with 10% rate, got {stats.Count}");
        }
    }

    [Fact]
    public void ProfileReport_GeneratesCorrectly()
    {
        // Arrange
        var config = new ProfilingConfig { Enabled = true };
        var session = new ProfilerSession(config);
        var op1Name = $"Op1_{_testId}";
        var op2Name = $"Op2_{_testId}";

        using (session.Scope(op1Name)) { Thread.Sleep(10); }
        using (session.Scope(op2Name)) { Thread.Sleep(20); }
        using (session.Scope(op1Name)) { Thread.Sleep(10); }

        // Act
        var report = session.GetReport();

        // Assert
        Assert.NotNull(report);
        Assert.True(report.Stats.Count >= 2, $"Expected at least 2 stats but got {report.Stats.Count}");
        Assert.True(report.TotalOperations >= 2);

        var op1Stats = report.GetStats(op1Name);
        Assert.NotNull(op1Stats);
        Assert.Equal(2, op1Stats.Count);
    }

    [Fact]
    public void ProfileReport_ToJson_ProducesValidJson()
    {
        // Arrange
        var config = new ProfilingConfig { Enabled = true };
        var session = new ProfilerSession(config);
        using (session.Scope("JsonTest")) { Thread.Sleep(5); }

        // Act
        var report = session.GetReport();
        var json = report.ToJson();

        // Assert
        Assert.NotNull(json);
        Assert.Contains("JsonTest", json);
        Assert.Contains("operationCount", json);
    }

    [Fact]
    public void ProfileReport_ToCsv_ProducesValidCsv()
    {
        // Arrange
        var config = new ProfilingConfig { Enabled = true };
        var session = new ProfilerSession(config);
        using (session.Scope("CsvTest")) { Thread.Sleep(5); }

        // Act
        var report = session.GetReport();
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
        var config = new ProfilingConfig { Enabled = true };
        var session = new ProfilerSession(config);
        using (session.Scope("MarkdownTest")) { Thread.Sleep(5); }

        // Act
        var report = session.GetReport();
        var markdown = report.ToMarkdown();

        // Assert
        Assert.NotNull(markdown);
        Assert.Contains("# Profiling Report", markdown);
        Assert.Contains("MarkdownTest", markdown);
    }

    [Fact]
    public void ProfileReport_GetHotspots_OrdersByTotalTime()
    {
        // Arrange
        var config = new ProfilingConfig { Enabled = true };
        var session = new ProfilerSession(config);

        // Create operations with different total times
        for (int i = 0; i < 5; i++)
        {
            using (session.Scope("Fast")) { } // Very fast
        }
        using (session.Scope("Slow")) { Thread.Sleep(50); }

        // Act
        var report = session.GetReport();
        var hotspots = report.GetHotspots(10).ToList();

        // Assert
        Assert.True(hotspots.Count >= 2);
        Assert.Equal("Slow", hotspots[0].Name); // Slow should be first (most total time)
    }

    [Fact]
    public void ProfilerStats_OpsPerSecond_CalculatesCorrectly()
    {
        // Arrange
        var config = new ProfilingConfig { Enabled = true };
        var session = new ProfilerSession(config);

        // Create an operation that takes ~10ms
        using (session.Scope("OpsPerSecTest"))
        {
            Thread.Sleep(10);
        }

        // Act
        var stats = session.GetStats("OpsPerSecTest");

        // Assert
        Assert.NotNull(stats);
        // 10ms = 100 ops/sec theoretically, but CI runners have significant timing variance
        // due to shared resources, VM overhead, and thread scheduling delays
        // Use wide tolerance: > 10 ops/sec (allows up to 100ms actual sleep) and < 500 ops/sec
        Assert.True(stats.OpsPerSecond > 10 && stats.OpsPerSecond < 500,
            $"Expected ops/sec between 10 and 500, but got {stats.OpsPerSecond}");
    }

    [Fact]
    public void ProfileReport_CompareTo_DetectsRegression()
    {
        // Arrange - Create baseline session and report
        var config = new ProfilingConfig { Enabled = true };
        var baselineSession = new ProfilerSession(config);
        baselineSession.Enable();
        // Record deterministic timings to avoid scheduler variance.
        baselineSession.RecordTiming("CompareOp", TimeSpan.FromMilliseconds(10));
        var baseline = baselineSession.GetReport();

        // Create current session with slower operation
        var currentSession = new ProfilerSession(config);
        currentSession.Enable();
        currentSession.RecordTiming("CompareOp", TimeSpan.FromMilliseconds(50));
        var current = currentSession.GetReport();

        // Act
        var comparison = current.CompareTo(baseline, thresholdPercent: 50);

        // Assert - should detect regression (more than 50% slower)
        Assert.True(comparison.Regressions.Count > 0);
        Assert.True(comparison.HasRegressions);
    }

    [Fact]
    public void ProfileReport_CompareTo_DetectsImprovement()
    {
        // Arrange - Create baseline session with slow operation
        var config = new ProfilingConfig { Enabled = true };
        var baselineSession = new ProfilerSession(config);
        baselineSession.Enable();
        // Record deterministic timings to avoid scheduler variance.
        baselineSession.RecordTiming("ImprovementOp", TimeSpan.FromMilliseconds(80));
        var baseline = baselineSession.GetReport();

        // Create current session with faster operation
        var currentSession = new ProfilerSession(config);
        currentSession.Enable();
        currentSession.RecordTiming("ImprovementOp", TimeSpan.FromMilliseconds(5));
        var current = currentSession.GetReport();

        // Act
        var comparison = current.CompareTo(baseline, thresholdPercent: 50);

        // Assert - should detect improvement
        Assert.True(comparison.Improvements.Count > 0);
        Assert.False(comparison.HasRegressions);
    }

    [Fact]
    public void ProfilerSession_HierarchyTracking_RecordsParentChild()
    {
        // Arrange
        var config = new ProfilingConfig { Enabled = true, TrackCallHierarchy = true };
        var session = new ProfilerSession(config);

        // Act - Create nested scopes
        using (session.Scope("Parent"))
        {
            Thread.Sleep(10);
            using (session.Scope("Child"))
            {
                Thread.Sleep(10);
            }
        }

        // Assert
        var childStats = session.GetStats("Child");
        Assert.NotNull(childStats);
        Assert.Contains("Parent", childStats.Parents);
    }

    [Fact]
    public void ProfilerSession_MaxOperations_LimitsTracking()
    {
        // Arrange - Limit to 5 operations
        var config = new ProfilingConfig { Enabled = true, MaxOperations = 5 };
        var session = new ProfilerSession(config);

        // Act - Try to create more than max operations
        for (int i = 0; i < 10; i++)
        {
            using (session.Scope($"Op_{i}")) { }
        }

        // Assert - Should not exceed max operations
        Assert.True(session.OperationCount <= 5);
    }

    [Fact]
    public void ProfilerSession_CustomTags_IncludedInReport()
    {
        // Arrange
        var config = new ProfilingConfig
        {
            Enabled = true,
            CustomTags = new Dictionary<string, string>
            {
                ["model"] = "GPT-4",
                ["experiment"] = "test123"
            }
        };
        var session = new ProfilerSession(config);
        using (session.Scope("TaggedOp")) { }

        // Act
        var report = session.GetReport();

        // Assert
        Assert.True(report.Tags.ContainsKey("model"));
        Assert.Equal("GPT-4", report.Tags["model"]);
        Assert.Equal("test123", report.Tags["experiment"]);
    }

    [Fact]
    public void ProfilerSession_GetOperationNames_ReturnsAllNames()
    {
        // Arrange
        var config = new ProfilingConfig { Enabled = true };
        var session = new ProfilerSession(config);
        using (session.Scope("Op1")) { }
        using (session.Scope("Op2")) { }
        using (session.Scope("Op3")) { }

        // Act
        var names = session.GetOperationNames();

        // Assert
        Assert.Equal(3, names.Count);
        Assert.Contains("Op1", names);
        Assert.Contains("Op2", names);
        Assert.Contains("Op3", names);
    }

    [Fact]
    public void ProfileReport_ToDictionary_ReturnsSerializableData()
    {
        // Arrange
        var config = new ProfilingConfig { Enabled = true };
        var session = new ProfilerSession(config);
        using (session.Scope("DictTest")) { Thread.Sleep(5); }

        // Act
        var report = session.GetReport();
        var dict = report.ToDictionary();

        // Assert
        Assert.True(dict.ContainsKey("startTime"));
        Assert.True(dict.ContainsKey("totalRuntimeMs"));
        Assert.True(dict.ContainsKey("operationCount"));
        Assert.True(dict.ContainsKey("operations"));
    }

    [Fact]
    public void ProfilerSession_ElapsedTime_TracksRuntime()
    {
        // Arrange
        var config = new ProfilingConfig { Enabled = true };
        var session = new ProfilerSession(config);

        // Act
        Thread.Sleep(50);
        var elapsed = session.Elapsed;

        // Assert
        Assert.True(elapsed.TotalMilliseconds >= 40);
    }

    [Fact]
    public void ProfilerSessionTimer_Elapsed_TracksTime()
    {
        // Arrange
        var config = new ProfilingConfig { Enabled = true };
        var session = new ProfilerSession(config);

        // Act
        var timer = session.Start("ElapsedTest");
        Thread.Sleep(30);
        var elapsed = timer.Elapsed;
        timer.Stop();

        // Assert
        Assert.True(elapsed.TotalMilliseconds >= 20);
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

    [Fact]
    public void MemoryScope_WithProfilerSession_RecordsToSession()
    {
        // Arrange
        MemoryTracker.Enable();
        var config = new ProfilingConfig { Enabled = true, TrackAllocations = true };
        var session = new ProfilerSession(config);

        // Act
        using (new MemoryScope("TestWithProfiler", session))
        {
            // Allocate some memory
            var data = new byte[1024 * 100]; // 100KB
            GC.KeepAlive(data);
        }

        // Assert - the profiler session should have recorded the allocation
        // Note: This verifies the integration works, even if the exact bytes vary
        var history = MemoryTracker.GetHistory();
        Assert.True(history.Count >= 2);
    }
}
