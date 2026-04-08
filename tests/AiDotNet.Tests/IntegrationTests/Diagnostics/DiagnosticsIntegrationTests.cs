using AiDotNet.Deployment.Configuration;
using AiDotNet.Diagnostics;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Diagnostics;

/// <summary>
/// Integration tests for the AiDotNet.Diagnostics module.
/// Tests MemoryTracker, ProfilerSession, and related diagnostic components.
/// </summary>
public class DiagnosticsIntegrationTests
{
    #region MemoryTracker Tests

    [Fact]
    public void MemoryTracker_Enable_SetsIsEnabled()
    {
        // Ensure clean state
        MemoryTracker.Disable();
        Assert.False(MemoryTracker.IsEnabled);

        MemoryTracker.Enable();
        Assert.True(MemoryTracker.IsEnabled);

        // Cleanup
        MemoryTracker.Disable();
    }

    [Fact]
    public void MemoryTracker_Disable_SetsIsEnabledFalse()
    {
        MemoryTracker.Enable();
        Assert.True(MemoryTracker.IsEnabled);

        MemoryTracker.Disable();
        Assert.False(MemoryTracker.IsEnabled);
    }

    [Fact]
    public void MemoryTracker_Snapshot_ReturnsValidSnapshot()
    {
        var snapshot = MemoryTracker.Snapshot("TestSnapshot");

        Assert.Equal("TestSnapshot", snapshot.Label);
        Assert.True(snapshot.TotalMemory > 0);
        Assert.True(snapshot.WorkingSet > 0);
        Assert.True(snapshot.HeapSizeBytes > 0);
        Assert.True(snapshot.Timestamp <= DateTime.UtcNow);
    }

    [Fact]
    public void MemoryTracker_Snapshot_WithForceGC_WorksCorrectly()
    {
        var snapshot = MemoryTracker.Snapshot("GCSnapshot", forceGC: true);

        Assert.Equal("GCSnapshot", snapshot.Label);
        Assert.True(snapshot.TotalMemory > 0);
    }

    [Fact]
    public void MemoryTracker_Snapshot_WithoutLabel_GeneratesDefaultLabel()
    {
        var snapshot = MemoryTracker.Snapshot();

        Assert.NotNull(snapshot.Label);
        Assert.StartsWith("Snapshot_", snapshot.Label);
    }

    [Fact]
    public void MemoryTracker_GetHistory_ReturnsRecordedSnapshots()
    {
        MemoryTracker.Reset();
        MemoryTracker.Enable();

        MemoryTracker.Snapshot("First");
        MemoryTracker.Snapshot("Second");

        var history = MemoryTracker.GetHistory();

        Assert.True(history.Count >= 2);
        Assert.Contains(history, s => s.Label == "First");
        Assert.Contains(history, s => s.Label == "Second");

        MemoryTracker.Disable();
        MemoryTracker.Reset();
    }

    [Fact]
    public void MemoryTracker_Reset_ClearsHistory()
    {
        MemoryTracker.Enable();
        MemoryTracker.Snapshot("BeforeReset");

        MemoryTracker.Reset();

        var history = MemoryTracker.GetHistory();
        Assert.Empty(history);

        MemoryTracker.Disable();
    }

    [Fact]
    public void MemoryTracker_GetPressureLevel_ReturnsValidLevel()
    {
        var level = MemoryTracker.GetPressureLevel();

        Assert.True(Enum.IsDefined(typeof(MemoryPressureLevel), level));
    }

    [Fact]
    public void MemoryTracker_EstimateTensorMemory_CalculatesCorrectly()
    {
        // 1000 x 1000 tensor with 4 bytes per element
        var memory = MemoryTracker.EstimateTensorMemory(new[] { 1000, 1000 }, elementSize: 4);

        Assert.Equal(4_000_000L, memory);
    }

    [Fact]
    public void MemoryTracker_EstimateTensorMemory_Handles3DTensor()
    {
        // 32 x 128 x 768 tensor with 4 bytes per element
        var memory = MemoryTracker.EstimateTensorMemory(new[] { 32, 128, 768 }, elementSize: 4);

        Assert.Equal(32L * 128 * 768 * 4, memory);
    }

    [Fact]
    public void MemoryTracker_EstimateKVCacheMemory_CalculatesCorrectly()
    {
        // 12 layers, 12 heads, 64 head dim, 512 seq len, batch 1, 4 bytes
        var memory = MemoryTracker.EstimateKVCacheMemory(
            numLayers: 12,
            numHeads: 12,
            headDim: 64,
            maxSeqLen: 512,
            batchSize: 1,
            bytesPerElement: 4);

        // K and V each: [1 * 12 * 512 * 64 * 4] * 2 = 3,145,728 per layer
        // Total: 3,145,728 * 12 = 37,748,736
        long expected = 1L * 12 * 512 * 64 * 4 * 2 * 12;
        Assert.Equal(expected, memory);
    }

    [Fact]
    public void MemoryTracker_TrackScope_ReturnsValidScope()
    {
        using var scope = MemoryTracker.TrackScope("ScopeTest");

        // Scope should have captured a before snapshot
        Assert.NotNull(scope.Before);
        Assert.Equal("ScopeTest_before", scope.Before.Label);
    }

    #endregion

    #region MemorySnapshot Tests

    [Fact]
    public void MemorySnapshot_CompareTo_ReturnsValidDiff()
    {
        var before = MemoryTracker.Snapshot("Before");

        // Allocate some memory
        var data = new byte[1024 * 1024]; // 1MB
        GC.KeepAlive(data);

        var after = MemoryTracker.Snapshot("After");
        var diff = after.CompareTo(before);

        Assert.Equal(before, diff.From);
        Assert.Equal(after, diff.To);
        Assert.True(diff.Duration.TotalMilliseconds >= 0);
    }

    [Fact]
    public void MemorySnapshot_ToString_ContainsLabel()
    {
        var snapshot = MemoryTracker.Snapshot("TestLabel");
        var str = snapshot.ToString();

        Assert.Contains("TestLabel", str);
        Assert.Contains("Total:", str);
    }

    [Fact]
    public void MemorySnapshot_Properties_ArePopulated()
    {
        var snapshot = MemoryTracker.Snapshot("PropsTest");

        // These should be populated with real values
        Assert.True(snapshot.Gen0Collections >= 0);
        Assert.True(snapshot.Gen1Collections >= 0);
        Assert.True(snapshot.Gen2Collections >= 0);
        Assert.True(snapshot.PrivateMemory > 0);
        Assert.True(snapshot.VirtualMemory > 0);
    }

    #endregion

    #region MemoryDiff Tests

    [Fact]
    public void MemoryDiff_AllocationRatePerSecond_CalculatesCorrectly()
    {
        var before = MemoryTracker.Snapshot("Before");
        Thread.Sleep(100);
        var after = MemoryTracker.Snapshot("After");
        var diff = after.CompareTo(before);

        // Allocation rate should be calculable (may be 0 if no significant allocation)
        Assert.True(diff.AllocationRatePerSecond >= 0 || diff.TotalMemoryDelta < 0);
    }

    [Fact]
    public void MemoryDiff_ToString_ContainsRelevantInfo()
    {
        var before = MemoryTracker.Snapshot("Before");
        var after = MemoryTracker.Snapshot("After");
        var diff = after.CompareTo(before);

        var str = diff.ToString();

        Assert.Contains("Memory delta:", str);
        Assert.Contains("GC:", str);
    }

    #endregion

    #region MemoryPressureLevel Tests

    [Fact]
    public void MemoryPressureLevel_AllValuesExist()
    {
        var values = (MemoryPressureLevel[])Enum.GetValues(typeof(MemoryPressureLevel));

        Assert.Contains(MemoryPressureLevel.Low, values);
        Assert.Contains(MemoryPressureLevel.Medium, values);
        Assert.Contains(MemoryPressureLevel.High, values);
        Assert.Contains(MemoryPressureLevel.Critical, values);
    }

    #endregion

    #region ProfilerSession Tests

    [Fact]
    public void ProfilerSession_DefaultConstructor_CreatesSession()
    {
        var session = new ProfilerSession();

        Assert.NotNull(session.Config);
        Assert.Equal(0, session.OperationCount);
    }

    [Fact]
    public void ProfilerSession_WithConfig_UsesConfig()
    {
        var config = new ProfilingConfig { Enabled = true, SamplingRate = 0.5 };
        var session = new ProfilerSession(config);

        Assert.Equal(config, session.Config);
        Assert.True(session.IsEnabled);
    }

    [Fact]
    public void ProfilerSession_Enable_SetsIsEnabled()
    {
        var session = new ProfilerSession(new ProfilingConfig { Enabled = false, AutoEnableInDebug = false });
        Assert.False(session.IsEnabled);

        session.Enable();
        Assert.True(session.IsEnabled);
    }

    [Fact]
    public void ProfilerSession_Disable_ClearsIsEnabled()
    {
        var session = new ProfilerSession(new ProfilingConfig { Enabled = true });
        Assert.True(session.IsEnabled);

        session.Disable();
        Assert.False(session.IsEnabled);
    }

    [Fact]
    public void ProfilerSession_Scope_TracksOperation()
    {
        var session = new ProfilerSession(new ProfilingConfig { Enabled = true });

        using (session.Scope("TestOperation"))
        {
            Thread.Sleep(10);
        }

        Assert.Equal(1, session.OperationCount);
        var stats = session.GetStats("TestOperation");
        Assert.NotNull(stats);
        Assert.Equal(1, stats.Count);
        Assert.True(stats.MeanMs >= 10);
    }

    [Fact]
    public void ProfilerSession_Start_CreatesManualTimer()
    {
        var session = new ProfilerSession(new ProfilingConfig { Enabled = true });

        var timer = session.Start("ManualOp");
        Thread.Sleep(10);
        timer.Stop();

        Assert.Equal(1, session.OperationCount);
        var stats = session.GetStats("ManualOp");
        Assert.NotNull(stats);
        Assert.True(stats.MeanMs >= 10);
    }

    [Fact]
    public void ProfilerSession_MultipleScopes_TracksAll()
    {
        var session = new ProfilerSession(new ProfilingConfig { Enabled = true });

        for (int i = 0; i < 5; i++)
        {
            using (session.Scope("RepeatedOp"))
            {
                Thread.Sleep(1);
            }
        }

        var stats = session.GetStats("RepeatedOp");
        Assert.NotNull(stats);
        Assert.Equal(5, stats.Count);
    }

    [Fact]
    public void ProfilerSession_Reset_ClearsData()
    {
        var session = new ProfilerSession(new ProfilingConfig { Enabled = true });

        using (session.Scope("BeforeReset"))
        {
            Thread.Sleep(1);
        }

        Assert.Equal(1, session.OperationCount);

        session.Reset();

        Assert.Equal(0, session.OperationCount);
    }

    [Fact]
    public void ProfilerSession_GetReport_ReturnsValidReport()
    {
        var session = new ProfilerSession(new ProfilingConfig { Enabled = true });

        using (session.Scope("Op1")) { Thread.Sleep(5); }
        using (session.Scope("Op2")) { Thread.Sleep(10); }

        var report = session.GetReport();

        Assert.NotNull(report);
        Assert.Equal(2, report.TotalOperations);
        Assert.True(report.TotalRuntime.TotalMilliseconds > 0);
    }

    [Fact]
    public void ProfilerSession_GetSummary_ReturnsString()
    {
        var session = new ProfilerSession(new ProfilingConfig { Enabled = true });

        using (session.Scope("SummaryOp")) { Thread.Sleep(1); }

        var summary = session.GetSummary();

        Assert.NotNull(summary);
        Assert.Contains("Profiling Report", summary);
    }

    [Fact]
    public void ProfilerSession_GetOperationNames_ReturnsNames()
    {
        var session = new ProfilerSession(new ProfilingConfig { Enabled = true });

        using (session.Scope("Alpha")) { }
        using (session.Scope("Beta")) { }
        using (session.Scope("Gamma")) { }

        var names = session.GetOperationNames();

        Assert.Equal(3, names.Count);
        Assert.Contains("Alpha", names);
        Assert.Contains("Beta", names);
        Assert.Contains("Gamma", names);
    }

    [Fact]
    public void ProfilerSession_Elapsed_TracksTime()
    {
        var session = new ProfilerSession();
        Thread.Sleep(50);

        Assert.True(session.Elapsed.TotalMilliseconds >= 50);
    }

    [Fact]
    public void ProfilerSession_DisabledSession_DoesNotRecord()
    {
        var session = new ProfilerSession(new ProfilingConfig { Enabled = false, AutoEnableInDebug = false });

        using (session.Scope("ShouldNotRecord"))
        {
            Thread.Sleep(1);
        }

        Assert.Equal(0, session.OperationCount);
    }

    #endregion

    #region ProfilerSessionTimer Tests

    [Fact]
    public void ProfilerSessionTimer_Name_ReturnsCorrectName()
    {
        var session = new ProfilerSession(new ProfilingConfig { Enabled = true });
        var timer = session.Start("TimerName");

        Assert.Equal("TimerName", timer.Name);

        timer.Stop();
    }

    [Fact]
    public void ProfilerSessionTimer_Elapsed_TracksTime()
    {
        var session = new ProfilerSession(new ProfilingConfig { Enabled = true });
        var timer = session.Start("ElapsedTest");

        Thread.Sleep(20);
        var elapsed = timer.Elapsed;

        Assert.True(elapsed.TotalMilliseconds >= 20);

        timer.Stop();
    }

    [Fact]
    public void ProfilerSessionTimer_DoubleStop_DoesNotThrow()
    {
        var session = new ProfilerSession(new ProfilingConfig { Enabled = true });
        var timer = session.Start("DoubleStop");

        timer.Stop();
        timer.Stop(); // Should not throw

        Assert.Equal(1, session.OperationCount);
    }

    [Fact]
    public void ProfilerSessionTimer_Dispose_StopsTimer()
    {
        var session = new ProfilerSession(new ProfilingConfig { Enabled = true });

        using (var timer = session.Start("DisposeTest"))
        {
            Thread.Sleep(5);
        }

        Assert.Equal(1, session.OperationCount);
    }

    #endregion

    #region ProfilerSessionScope Tests

    [Fact]
    public void ProfilerSessionScope_Elapsed_TracksTime()
    {
        var session = new ProfilerSession(new ProfilingConfig { Enabled = true });

        using (var scope = session.Scope("ScopeElapsed"))
        {
            Thread.Sleep(15);
            Assert.True(scope.Elapsed.TotalMilliseconds >= 15);
        }
    }

    #endregion

    #region ProfileReport Tests

    [Fact]
    public void ProfileReport_GetAllStats_ReturnsSortedByTotalTime()
    {
        var session = new ProfilerSession(new ProfilingConfig { Enabled = true });

        using (session.Scope("Fast")) { Thread.Sleep(5); }
        using (session.Scope("Slow")) { Thread.Sleep(20); }
        using (session.Scope("Medium")) { Thread.Sleep(10); }

        var report = session.GetReport();
        var allStats = report.GetAllStats();

        Assert.Equal(3, allStats.Count);
        Assert.Equal("Slow", allStats[0].Name);
    }

    [Fact]
    public void ProfileReport_GetTopOperations_ReturnsLimitedCount()
    {
        var session = new ProfilerSession(new ProfilingConfig { Enabled = true });

        for (int i = 0; i < 20; i++)
        {
            using (session.Scope($"Op{i}")) { }
        }

        var report = session.GetReport();
        var top5 = report.GetTopOperations(5);

        Assert.Equal(5, top5.Count);
    }

    [Fact]
    public void ProfileReport_GetSlowOperations_FiltersCorrectly()
    {
        var session = new ProfilerSession(new ProfilingConfig { Enabled = true });

        using (session.Scope("Fast")) { Thread.Sleep(1); }
        using (session.Scope("Slow")) { Thread.Sleep(50); }

        var report = session.GetReport();
        var slowOps = report.GetSlowOperations(p95ThresholdMs: 30);

        Assert.Single(slowOps);
        Assert.Equal("Slow", slowOps[0].Name);
    }

    [Fact]
    public void ProfileReport_GetHotspots_ReturnsHottestOperations()
    {
        var session = new ProfilerSession(new ProfilingConfig { Enabled = true });

        using (session.Scope("Cold")) { Thread.Sleep(1); }
        using (session.Scope("Hot")) { Thread.Sleep(30); }
        using (session.Scope("Warm")) { Thread.Sleep(10); }

        var report = session.GetReport();
        var hotspots = report.GetHotspots(2);

        Assert.Equal(2, hotspots.Count);
        Assert.Equal("Hot", hotspots[0].Name);
    }

    [Fact]
    public void ProfileReport_ToJson_ReturnsValidJson()
    {
        var session = new ProfilerSession(new ProfilingConfig { Enabled = true });
        using (session.Scope("JsonOp")) { Thread.Sleep(1); }

        var report = session.GetReport();
        var json = report.ToJson();

        Assert.Contains("\"startTime\"", json);
        Assert.Contains("\"operations\"", json);
        Assert.Contains("\"JsonOp\"", json);
    }

    [Fact]
    public void ProfileReport_ToCsv_ReturnsValidCsv()
    {
        var session = new ProfilerSession(new ProfilingConfig { Enabled = true });
        using (session.Scope("CsvOp")) { Thread.Sleep(1); }

        var report = session.GetReport();
        var csv = report.ToCsv();

        Assert.Contains("Name,Count,TotalMs", csv);
        Assert.Contains("CsvOp", csv);
    }

    [Fact]
    public void ProfileReport_ToMarkdown_ReturnsValidMarkdown()
    {
        var session = new ProfilerSession(new ProfilingConfig { Enabled = true });
        using (session.Scope("MarkdownOp")) { Thread.Sleep(1); }

        var report = session.GetReport();
        var md = report.ToMarkdown();

        Assert.Contains("# Profiling Report", md);
        Assert.Contains("| Operation |", md);
        Assert.Contains("MarkdownOp", md);
    }

    [Fact]
    public void ProfileReport_ToDictionary_ReturnsValidDictionary()
    {
        var session = new ProfilerSession(new ProfilingConfig { Enabled = true });
        using (session.Scope("DictOp")) { Thread.Sleep(1); }

        var report = session.GetReport();
        var dict = report.ToDictionary();

        Assert.Contains("startTime", dict.Keys);
        Assert.Contains("totalRuntimeMs", dict.Keys);
        Assert.Contains("operationCount", dict.Keys);
        Assert.Contains("operations", dict.Keys);
    }

    [Fact]
    public void ProfileReport_GetStats_ReturnsStatsForOperation()
    {
        var session = new ProfilerSession(new ProfilingConfig { Enabled = true });
        using (session.Scope("SpecificOp")) { Thread.Sleep(5); }

        var report = session.GetReport();
        var stats = report.GetStats("SpecificOp");

        Assert.NotNull(stats);
        Assert.Equal("SpecificOp", stats.Name);
    }

    [Fact]
    public void ProfileReport_GetStats_ReturnsNullForUnknownOperation()
    {
        var session = new ProfilerSession(new ProfilingConfig { Enabled = true });
        var report = session.GetReport();

        var stats = report.GetStats("NonExistent");

        Assert.Null(stats);
    }

    [Fact]
    public void ProfileReport_CompareTo_DetectsRegressions()
    {
        var session1 = new ProfilerSession(new ProfilingConfig { Enabled = true });
        using (session1.Scope("CompareOp")) { Thread.Sleep(10); }
        var baseline = session1.GetReport();

        var session2 = new ProfilerSession(new ProfilingConfig { Enabled = true });
        using (session2.Scope("CompareOp")) { Thread.Sleep(50); }
        var current = session2.GetReport();

        var comparison = current.CompareTo(baseline, thresholdPercent: 50);

        Assert.True(comparison.HasRegressions);
        Assert.Single(comparison.Regressions);
        Assert.Equal("CompareOp", comparison.Regressions[0].OperationName);
    }

    [Fact]
    public void ProfileReport_CompareTo_DetectsImprovements()
    {
        var session1 = new ProfilerSession(new ProfilingConfig { Enabled = true });
        using (session1.Scope("ImproveOp")) { Thread.Sleep(50); }
        var baseline = session1.GetReport();

        var session2 = new ProfilerSession(new ProfilingConfig { Enabled = true });
        using (session2.Scope("ImproveOp")) { Thread.Sleep(10); }
        var current = session2.GetReport();

        var comparison = current.CompareTo(baseline, thresholdPercent: 20);

        Assert.True(comparison.Improvements.Count > 0);
    }

    [Fact]
    public void ProfileReport_ToString_ContainsReportInfo()
    {
        var session = new ProfilerSession(new ProfilingConfig { Enabled = true });
        using (session.Scope("ToStringOp")) { Thread.Sleep(1); }

        var report = session.GetReport();
        var str = report.ToString();

        Assert.Contains("Profiling Report", str);
        Assert.Contains("Operations Tracked", str);
    }

    #endregion

    #region ProfilerStats Tests

    [Fact]
    public void ProfilerStats_OpsPerSecond_CalculatesCorrectly()
    {
        var session = new ProfilerSession(new ProfilingConfig { Enabled = true });

        for (int i = 0; i < 10; i++)
        {
            using (session.Scope("FrequentOp")) { }
        }

        var stats = session.GetStats("FrequentOp");
        Assert.NotNull(stats);
        Assert.True(stats.OpsPerSecond > 0);
    }

    [Fact]
    public void ProfilerStats_ToString_ContainsName()
    {
        var session = new ProfilerSession(new ProfilingConfig { Enabled = true });
        using (session.Scope("StatsToString")) { Thread.Sleep(1); }

        var stats = session.GetStats("StatsToString");
        Assert.NotNull(stats);

        var str = stats.ToString();
        Assert.Contains("StatsToString", str);
        Assert.Contains("calls", str);
    }

    [Fact]
    public void ProfilerStats_Percentiles_AreCalculated()
    {
        var session = new ProfilerSession(new ProfilingConfig { Enabled = true });

        for (int i = 0; i < 100; i++)
        {
            using (session.Scope("PercentileOp"))
            {
                Thread.Sleep(1);
            }
        }

        var stats = session.GetStats("PercentileOp");
        Assert.NotNull(stats);
        Assert.True(stats.P50Ms > 0);
        Assert.True(stats.P95Ms >= stats.P50Ms);
        Assert.True(stats.P99Ms >= stats.P95Ms);
    }

    [Fact]
    public void ProfilerStats_MinMax_AreTracked()
    {
        var session = new ProfilerSession(new ProfilingConfig { Enabled = true });

        using (session.Scope("MinMaxOp")) { Thread.Sleep(5); }
        using (session.Scope("MinMaxOp")) { Thread.Sleep(20); }
        using (session.Scope("MinMaxOp")) { Thread.Sleep(10); }

        var stats = session.GetStats("MinMaxOp");
        Assert.NotNull(stats);
        Assert.True(stats.MinMs >= 5);
        Assert.True(stats.MaxMs >= 20);
        Assert.True(stats.MinMs <= stats.MaxMs);
    }

    #endregion

    #region ProfileComparison Tests

    [Fact]
    public void ProfileComparison_ToString_ContainsComparisonInfo()
    {
        var session1 = new ProfilerSession(new ProfilingConfig { Enabled = true });
        using (session1.Scope("CompOp")) { Thread.Sleep(10); }
        var baseline = session1.GetReport();

        var session2 = new ProfilerSession(new ProfilingConfig { Enabled = true });
        using (session2.Scope("CompOp")) { Thread.Sleep(30); }
        var current = session2.GetReport();

        var comparison = current.CompareTo(baseline, thresholdPercent: 10);
        var str = comparison.ToString();

        Assert.Contains("Profile Comparison", str);
        Assert.Contains("threshold", str);
    }

    #endregion

    #region ProfilingConfig Tests

    [Fact]
    public void ProfilingConfig_DefaultValues_AreSet()
    {
        var config = new ProfilingConfig();

        Assert.False(config.Enabled);
        Assert.Equal(1.0, config.SamplingRate);
        Assert.Equal(1000, config.ReservoirSize);
        Assert.Equal(10000, config.MaxOperations);
        Assert.True(config.TrackAllocations);
        Assert.True(config.TrackCallHierarchy);
    }

    [Fact]
    public void ProfilingConfig_CustomValues_ArePreserved()
    {
        var config = new ProfilingConfig
        {
            Enabled = true,
            SamplingRate = 0.5,
            ReservoirSize = 500,
            MaxOperations = 5000,
            TrackAllocations = false,
            TrackCallHierarchy = false
        };

        Assert.True(config.Enabled);
        Assert.Equal(0.5, config.SamplingRate);
        Assert.Equal(500, config.ReservoirSize);
        Assert.Equal(5000, config.MaxOperations);
        Assert.False(config.TrackAllocations);
        Assert.False(config.TrackCallHierarchy);
    }

    #endregion

    #region Integration Scenarios

    [Fact]
    public void Integration_MemoryAndProfiler_WorkTogether()
    {
        var session = new ProfilerSession(new ProfilingConfig
        {
            Enabled = true,
            TrackAllocations = true
        });

        using (session.Scope("MemoryIntensiveOp"))
        {
            using var memScope = MemoryTracker.TrackScope("AllocationTracking");
            var data = new byte[1024 * 100]; // 100KB
            GC.KeepAlive(data);
        }

        var report = session.GetReport();
        Assert.Equal(1, report.TotalOperations);
    }

    [Fact]
    public void Integration_NestedScopes_WorkCorrectly()
    {
        var session = new ProfilerSession(new ProfilingConfig { Enabled = true });

        using (session.Scope("Outer"))
        {
            Thread.Sleep(5);
            using (session.Scope("Inner"))
            {
                Thread.Sleep(10);
            }
        }

        var outerStats = session.GetStats("Outer");
        var innerStats = session.GetStats("Inner");

        Assert.NotNull(outerStats);
        Assert.NotNull(innerStats);
        Assert.True(outerStats.TotalMs >= innerStats.TotalMs);
    }

    [Fact]
    public void Integration_SamplingRate_AffectsRecording()
    {
        var session = new ProfilerSession(new ProfilingConfig
        {
            Enabled = true,
            SamplingRate = 0.0 // Sample nothing
        });

        for (int i = 0; i < 100; i++)
        {
            using (session.Scope("SampledOp")) { }
        }

        // With 0% sampling rate, nothing should be recorded
        Assert.Equal(0, session.OperationCount);
    }

    [Fact]
    public void Integration_ConcurrentProfiling_IsThreadSafe()
    {
        var session = new ProfilerSession(new ProfilingConfig { Enabled = true });
        var tasks = new List<Task>();

        for (int i = 0; i < 10; i++)
        {
            int threadId = i;
            tasks.Add(Task.Run(() =>
            {
                for (int j = 0; j < 10; j++)
                {
                    using (session.Scope($"Thread{threadId}Op"))
                    {
                        Thread.Sleep(1);
                    }
                }
            }));
        }

        Task.WaitAll(tasks.ToArray());

        Assert.Equal(10, session.OperationCount);
        var report = session.GetReport();
        Assert.True(report.GetAllStats().Sum(s => s.Count) >= 100);
    }

    #endregion
}
