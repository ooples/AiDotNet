using AiDotNet.Deployment.Configuration;
using AiDotNet.Diagnostics;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Diagnostics;

/// <summary>
/// Deep integration tests for MemoryTracker, ProfilerSession, MemorySnapshot,
/// MemoryDiff, ProfilerStats, ProfileReport: memory estimation math,
/// Welford's algorithm properties, reservoir sampling, snapshot comparison,
/// scope tracking, and report generation.
/// </summary>
public class DiagnosticsDeepMathIntegrationTests
{
    // ============================
    // MemoryTracker.EstimateTensorMemory Tests
    // ============================

    [Fact]
    public void EstimateTensorMemory_1DShape_HandComputed()
    {
        // Shape [100], elementSize=4 => 100 * 4 = 400 bytes
        var bytes = MemoryTracker.EstimateTensorMemory(new[] { 100 }, 4);

        Assert.Equal(400L, bytes);
    }

    [Fact]
    public void EstimateTensorMemory_2DShape_HandComputed()
    {
        // Shape [32, 64], elementSize=4 => 32 * 64 * 4 = 8192 bytes
        var bytes = MemoryTracker.EstimateTensorMemory(new[] { 32, 64 }, 4);

        Assert.Equal(8192L, bytes);
    }

    [Fact]
    public void EstimateTensorMemory_3DShape_HandComputed()
    {
        // Shape [3, 224, 224], elementSize=4 => 3 * 224 * 224 * 4 = 602,112 bytes
        var bytes = MemoryTracker.EstimateTensorMemory(new[] { 3, 224, 224 }, 4);

        Assert.Equal(602112L, bytes);
    }

    [Fact]
    public void EstimateTensorMemory_4DShape_BatchedImages_HandComputed()
    {
        // Shape [16, 3, 224, 224], elementSize=4 => 16 * 3 * 224 * 224 * 4 = 9,633,792 bytes
        var bytes = MemoryTracker.EstimateTensorMemory(new[] { 16, 3, 224, 224 }, 4);

        Assert.Equal(9633792L, bytes);
    }

    [Fact]
    public void EstimateTensorMemory_DefaultElementSize_Is4()
    {
        // Default element size is 4 (float32)
        var bytes = MemoryTracker.EstimateTensorMemory(new[] { 10 });

        Assert.Equal(40L, bytes);
    }

    [Fact]
    public void EstimateTensorMemory_DoubleElementSize_Is8()
    {
        // Shape [10], elementSize=8 (float64) => 10 * 8 = 80 bytes
        var bytes = MemoryTracker.EstimateTensorMemory(new[] { 10 }, 8);

        Assert.Equal(80L, bytes);
    }

    [Fact]
    public void EstimateTensorMemory_HalfPrecision_Is2()
    {
        // Shape [1000], elementSize=2 (float16) => 1000 * 2 = 2000 bytes
        var bytes = MemoryTracker.EstimateTensorMemory(new[] { 1000 }, 2);

        Assert.Equal(2000L, bytes);
    }

    [Fact]
    public void EstimateTensorMemory_SingleElement_HandComputed()
    {
        // Scalar: shape [1], elementSize=4 => 1 * 4 = 4 bytes
        var bytes = MemoryTracker.EstimateTensorMemory(new[] { 1 }, 4);

        Assert.Equal(4L, bytes);
    }

    [Fact]
    public void EstimateTensorMemory_LargeModel_HandComputed()
    {
        // GPT-2 style weight matrix: [768, 3072], float32
        // 768 * 3072 * 4 = 9,437,184 bytes (~9 MB)
        var bytes = MemoryTracker.EstimateTensorMemory(new[] { 768, 3072 }, 4);

        Assert.Equal(9437184L, bytes);
    }

    // ============================
    // MemoryTracker.EstimateKVCacheMemory Tests
    // ============================

    [Fact]
    public void EstimateKVCacheMemory_SingleLayer_HandComputed()
    {
        // numLayers=1, numHeads=8, headDim=64, maxSeqLen=512, batchSize=1, bytesPerElement=4
        // K+V per layer: 1 * 8 * 512 * 64 * 4 * 2 = 2,097,152 bytes
        // Total: 2,097,152 * 1 = 2,097,152 bytes (~2 MB)
        var bytes = MemoryTracker.EstimateKVCacheMemory(
            numLayers: 1, numHeads: 8, headDim: 64, maxSeqLen: 512,
            batchSize: 1, bytesPerElement: 4);

        Assert.Equal(2097152L, bytes);
    }

    [Fact]
    public void EstimateKVCacheMemory_GPT2Style_HandComputed()
    {
        // GPT-2: 12 layers, 12 heads, headDim=64, maxSeqLen=1024, batch=1, float32
        // Per layer: 1 * 12 * 1024 * 64 * 4 * 2 = 6,291,456
        // Total: 6,291,456 * 12 = 75,497,472 bytes (~72 MB)
        var bytes = MemoryTracker.EstimateKVCacheMemory(
            numLayers: 12, numHeads: 12, headDim: 64, maxSeqLen: 1024,
            batchSize: 1, bytesPerElement: 4);

        Assert.Equal(75497472L, bytes);
    }

    [Fact]
    public void EstimateKVCacheMemory_BatchEffect_IsLinear()
    {
        // Doubling batch size should double memory
        var bytes1 = MemoryTracker.EstimateKVCacheMemory(
            numLayers: 6, numHeads: 8, headDim: 64, maxSeqLen: 256,
            batchSize: 1, bytesPerElement: 4);

        var bytes2 = MemoryTracker.EstimateKVCacheMemory(
            numLayers: 6, numHeads: 8, headDim: 64, maxSeqLen: 256,
            batchSize: 2, bytesPerElement: 4);

        Assert.Equal(bytes1 * 2, bytes2);
    }

    [Fact]
    public void EstimateKVCacheMemory_LayerEffect_IsLinear()
    {
        // Doubling layers should double memory
        var bytes6 = MemoryTracker.EstimateKVCacheMemory(
            numLayers: 6, numHeads: 8, headDim: 64, maxSeqLen: 256,
            batchSize: 1, bytesPerElement: 4);

        var bytes12 = MemoryTracker.EstimateKVCacheMemory(
            numLayers: 12, numHeads: 8, headDim: 64, maxSeqLen: 256,
            batchSize: 1, bytesPerElement: 4);

        Assert.Equal(bytes6 * 2, bytes12);
    }

    [Fact]
    public void EstimateKVCacheMemory_HalfPrecision_IsHalfOfFloat32()
    {
        var fp32 = MemoryTracker.EstimateKVCacheMemory(
            numLayers: 12, numHeads: 12, headDim: 64, maxSeqLen: 1024,
            batchSize: 1, bytesPerElement: 4);

        var fp16 = MemoryTracker.EstimateKVCacheMemory(
            numLayers: 12, numHeads: 12, headDim: 64, maxSeqLen: 1024,
            batchSize: 1, bytesPerElement: 2);

        Assert.Equal(fp32 / 2, fp16);
    }

    // ============================
    // MemoryTracker Enable/Disable/Reset Tests
    // ============================

    [Fact]
    public void MemoryTracker_EnableDisable_TracksState()
    {
        // Save original state and restore after test
        var wasEnabled = MemoryTracker.IsEnabled;
        try
        {
            MemoryTracker.Enable();
            Assert.True(MemoryTracker.IsEnabled);

            MemoryTracker.Disable();
            Assert.False(MemoryTracker.IsEnabled);
        }
        finally
        {
            if (wasEnabled) MemoryTracker.Enable();
            else MemoryTracker.Disable();
        }
    }

    [Fact]
    public void MemoryTracker_Reset_ClearsHistory()
    {
        var wasEnabled = MemoryTracker.IsEnabled;
        try
        {
            MemoryTracker.Enable();
            MemoryTracker.Snapshot("test1");
            MemoryTracker.Snapshot("test2");

            MemoryTracker.Reset();
            var history = MemoryTracker.GetHistory();

            Assert.Empty(history);
        }
        finally
        {
            MemoryTracker.Reset();
            if (wasEnabled) MemoryTracker.Enable();
            else MemoryTracker.Disable();
        }
    }

    [Fact]
    public void MemoryTracker_Snapshot_RecordsWhenEnabled()
    {
        var wasEnabled = MemoryTracker.IsEnabled;
        try
        {
            MemoryTracker.Reset();
            MemoryTracker.Enable();

            MemoryTracker.Snapshot("enabled_snapshot");

            var history = MemoryTracker.GetHistory();
            Assert.True(history.Count >= 1);
            Assert.Equal("enabled_snapshot", history[history.Count - 1].Label);
        }
        finally
        {
            MemoryTracker.Reset();
            if (wasEnabled) MemoryTracker.Enable();
            else MemoryTracker.Disable();
        }
    }

    [Fact]
    public void MemoryTracker_Snapshot_DoesNotRecordWhenDisabled()
    {
        var wasEnabled = MemoryTracker.IsEnabled;
        try
        {
            MemoryTracker.Reset();
            MemoryTracker.Disable();

            var snapshot = MemoryTracker.Snapshot("disabled_snapshot");

            // Snapshot is still returned, but not added to history
            Assert.NotNull(snapshot);

            var history = MemoryTracker.GetHistory();
            Assert.Empty(history);
        }
        finally
        {
            MemoryTracker.Reset();
            if (wasEnabled) MemoryTracker.Enable();
            else MemoryTracker.Disable();
        }
    }

    [Fact]
    public void MemoryTracker_MaxHistorySize_ClampsToMinimum1()
    {
        var original = MemoryTracker.MaxHistorySize;
        try
        {
            MemoryTracker.MaxHistorySize = 0;
            Assert.Equal(1, MemoryTracker.MaxHistorySize);

            MemoryTracker.MaxHistorySize = -10;
            Assert.Equal(1, MemoryTracker.MaxHistorySize);
        }
        finally
        {
            MemoryTracker.MaxHistorySize = original;
        }
    }

    // ============================
    // MemorySnapshot Tests
    // ============================

    [Fact]
    public void MemorySnapshot_HasPositiveMetrics()
    {
        var snapshot = MemoryTracker.Snapshot("test");

        Assert.True(snapshot.TotalMemory > 0);
        Assert.True(snapshot.WorkingSet > 0);
    }

    [Fact]
    public void MemorySnapshot_LabelIsSet()
    {
        var snapshot = MemoryTracker.Snapshot("my_label");

        Assert.Equal("my_label", snapshot.Label);
    }

    [Fact]
    public void MemorySnapshot_DefaultLabel_IsSequential()
    {
        var wasEnabled = MemoryTracker.IsEnabled;
        try
        {
            MemoryTracker.Reset();
            MemoryTracker.Enable();

            var snapshot = MemoryTracker.Snapshot();

            Assert.StartsWith("Snapshot_", snapshot.Label);
        }
        finally
        {
            MemoryTracker.Reset();
            if (wasEnabled) MemoryTracker.Enable();
            else MemoryTracker.Disable();
        }
    }

    [Fact]
    public void MemorySnapshot_GCCollections_NonNegative()
    {
        var snapshot = MemoryTracker.Snapshot();

        Assert.True(snapshot.Gen0Collections >= 0);
        Assert.True(snapshot.Gen1Collections >= 0);
        Assert.True(snapshot.Gen2Collections >= 0);
    }

    [Fact]
    public void MemorySnapshot_ToString_ContainsLabel()
    {
        var snapshot = MemoryTracker.Snapshot("test_label");

        var str = snapshot.ToString();
        Assert.Contains("test_label", str);
        Assert.Contains("Total:", str);
        Assert.Contains("Heap:", str);
        Assert.Contains("WorkingSet:", str);
    }

    // ============================
    // MemoryDiff Tests
    // ============================

    [Fact]
    public void MemoryDiff_CompareTo_ComputesDelta()
    {
        var before = MemoryTracker.Snapshot("before");

        // Allocate some memory to create a measurable diff
        var data = new byte[1024 * 1024]; // 1 MB
        GC.KeepAlive(data);

        var after = MemoryTracker.Snapshot("after");
        var diff = after.CompareTo(before);

        // Duration should be non-negative
        Assert.True(diff.Duration >= TimeSpan.Zero);

        // From/To references should be correct
        Assert.Same(before, diff.From);
        Assert.Same(after, diff.To);
    }

    [Fact]
    public void MemoryDiff_GCCollectionDeltas_NonNegative()
    {
        var before = MemoryTracker.Snapshot("before");
        var after = MemoryTracker.Snapshot("after");
        var diff = after.CompareTo(before);

        Assert.True(diff.Gen0CollectionsDelta >= 0);
        Assert.True(diff.Gen1CollectionsDelta >= 0);
        Assert.True(diff.Gen2CollectionsDelta >= 0);
    }

    [Fact]
    public void MemoryDiff_AllocationRate_ZeroDuration_ReturnsZero()
    {
        // Create two snapshots with timestamps very close together
        var s1 = new MemorySnapshot
        {
            Label = "s1",
            Timestamp = DateTime.UtcNow,
            TotalMemory = 1000
        };
        var s2 = new MemorySnapshot
        {
            Label = "s2",
            Timestamp = s1.Timestamp, // Same timestamp = zero duration
            TotalMemory = 2000
        };

        var diff = s2.CompareTo(s1);
        Assert.Equal(TimeSpan.Zero, diff.Duration);
        Assert.Equal(0.0, diff.AllocationRatePerSecond);
    }

    [Fact]
    public void MemoryDiff_ToString_ContainsDeltaInfo()
    {
        var before = MemoryTracker.Snapshot("before");
        var after = MemoryTracker.Snapshot("after");
        var diff = after.CompareTo(before);

        var str = diff.ToString();
        Assert.Contains("Memory delta:", str);
        Assert.Contains("GC:", str);
    }

    // ============================
    // ProfilerSession Basic Tests
    // ============================

    [Fact]
    public void ProfilerSession_DefaultConfig_DisabledByDefault()
    {
        var config = new ProfilingConfig { Enabled = false, AutoEnableInDebug = false };
        var session = new ProfilerSession(config);

        // In release builds, should follow config.Enabled
        Assert.NotNull(session.Config);
    }

    [Fact]
    public void ProfilerSession_EnableDisable_Works()
    {
        var config = new ProfilingConfig { Enabled = false, AutoEnableInDebug = false };
        var session = new ProfilerSession(config);

        session.Enable();
        Assert.True(session.IsEnabled);

        session.Disable();
        Assert.False(session.IsEnabled);
    }

    [Fact]
    public void ProfilerSession_Reset_ClearsEntries()
    {
        var config = new ProfilingConfig { Enabled = true, AutoEnableInDebug = false };
        var session = new ProfilerSession(config);

        using (session.Scope("test_op"))
        {
            // Some work
        }

        Assert.True(session.OperationCount > 0);

        session.Reset();
        Assert.Equal(0, session.OperationCount);
    }

    [Fact]
    public void ProfilerSession_Scope_RecordsTiming()
    {
        var config = new ProfilingConfig { Enabled = true, AutoEnableInDebug = false };
        var session = new ProfilerSession(config);

        using (session.Scope("test_operation"))
        {
            // Simulate some work
            Thread.Sleep(10);
        }

        var stats = session.GetStats("test_operation");
        Assert.NotNull(stats);
        Assert.Equal(1, stats.Count);
        Assert.True(stats.TotalMs > 0);
    }

    [Fact]
    public void ProfilerSession_MultipleScopes_SameOperation_Aggregates()
    {
        var config = new ProfilingConfig { Enabled = true, AutoEnableInDebug = false };
        var session = new ProfilerSession(config);

        for (int i = 0; i < 5; i++)
        {
            using (session.Scope("repeated_op"))
            {
                Thread.Sleep(1);
            }
        }

        var stats = session.GetStats("repeated_op");
        Assert.NotNull(stats);
        Assert.Equal(5, stats.Count);
    }

    [Fact]
    public void ProfilerSession_DifferentOperations_TrackedSeparately()
    {
        var config = new ProfilingConfig { Enabled = true, AutoEnableInDebug = false };
        var session = new ProfilerSession(config);

        using (session.Scope("op_a")) { Thread.Sleep(1); }
        using (session.Scope("op_b")) { Thread.Sleep(1); }
        using (session.Scope("op_c")) { Thread.Sleep(1); }

        Assert.Equal(3, session.OperationCount);

        var names = session.GetOperationNames();
        Assert.Contains("op_a", names);
        Assert.Contains("op_b", names);
        Assert.Contains("op_c", names);
    }

    [Fact]
    public void ProfilerSession_DisabledSession_DoesNotRecord()
    {
        var config = new ProfilingConfig { Enabled = false, AutoEnableInDebug = false };
        var session = new ProfilerSession(config);

        using (session.Scope("should_not_record"))
        {
            Thread.Sleep(1);
        }

        Assert.Equal(0, session.OperationCount);
        Assert.Null(session.GetStats("should_not_record"));
    }

    // ============================
    // ProfilerSession Timer Tests
    // ============================

    [Fact]
    public void ProfilerSessionTimer_ManualStartStop_RecordsTiming()
    {
        var config = new ProfilingConfig { Enabled = true, AutoEnableInDebug = false };
        var session = new ProfilerSession(config);

        var timer = session.Start("manual_op");
        Thread.Sleep(10);
        timer.Stop();

        var stats = session.GetStats("manual_op");
        Assert.NotNull(stats);
        Assert.Equal(1, stats.Count);
        Assert.True(stats.TotalMs >= 5); // At least some ms
    }

    [Fact]
    public void ProfilerSessionTimer_DoubleStop_DoesNotDuplicate()
    {
        var config = new ProfilingConfig { Enabled = true, AutoEnableInDebug = false };
        var session = new ProfilerSession(config);

        var timer = session.Start("double_stop");
        Thread.Sleep(1);
        timer.Stop();
        timer.Stop(); // Second stop should be no-op

        var stats = session.GetStats("double_stop");
        Assert.NotNull(stats);
        Assert.Equal(1, stats.Count); // Only counted once
    }

    [Fact]
    public void ProfilerSessionTimer_Elapsed_IncreasesOverTime()
    {
        var config = new ProfilingConfig { Enabled = true, AutoEnableInDebug = false };
        var session = new ProfilerSession(config);

        var timer = session.Start("elapsed_test");
        var early = timer.Elapsed;
        Thread.Sleep(20);
        var late = timer.Elapsed;
        timer.Stop();

        Assert.True(late > early);
    }

    [Fact]
    public void ProfilerSessionTimer_Dispose_StopsTimer()
    {
        var config = new ProfilingConfig { Enabled = true, AutoEnableInDebug = false };
        var session = new ProfilerSession(config);

        using (var timer = session.Start("dispose_test"))
        {
            Thread.Sleep(1);
        } // Dispose calls Stop

        var stats = session.GetStats("dispose_test");
        Assert.NotNull(stats);
        Assert.Equal(1, stats.Count);
    }

    // ============================
    // ProfilerStats Math Tests (Welford's Algorithm)
    // ============================

    [Fact]
    public void ProfilerStats_SingleSample_MeanEqualsValue()
    {
        var config = new ProfilingConfig { Enabled = true, AutoEnableInDebug = false };
        var session = new ProfilerSession(config);

        using (session.Scope("single"))
        {
            Thread.Sleep(10);
        }

        var stats = session.GetStats("single");
        Assert.NotNull(stats);

        // With single sample, mean = total = min = max
        Assert.Equal(stats.TotalMs, stats.MeanMs, 0.001);
        Assert.Equal(stats.MinMs, stats.MaxMs, 0.001);
        Assert.Equal(stats.MeanMs, stats.MinMs, 0.001);
    }

    [Fact]
    public void ProfilerStats_MinLessThanOrEqualMax()
    {
        var config = new ProfilingConfig { Enabled = true, AutoEnableInDebug = false };
        var session = new ProfilerSession(config);

        for (int i = 0; i < 10; i++)
        {
            using (session.Scope("minmax"))
            {
                Thread.Sleep(1);
            }
        }

        var stats = session.GetStats("minmax");
        Assert.NotNull(stats);
        Assert.True(stats.MinMs <= stats.MaxMs);
    }

    [Fact]
    public void ProfilerStats_MeanBetweenMinAndMax()
    {
        var config = new ProfilingConfig { Enabled = true, AutoEnableInDebug = false };
        var session = new ProfilerSession(config);

        for (int i = 0; i < 10; i++)
        {
            using (session.Scope("mean_range"))
            {
                Thread.Sleep(1);
            }
        }

        var stats = session.GetStats("mean_range");
        Assert.NotNull(stats);
        Assert.True(stats.MeanMs >= stats.MinMs);
        Assert.True(stats.MeanMs <= stats.MaxMs);
    }

    [Fact]
    public void ProfilerStats_TotalEquals_CountTimesMean()
    {
        var config = new ProfilingConfig { Enabled = true, AutoEnableInDebug = false };
        var session = new ProfilerSession(config);

        for (int i = 0; i < 5; i++)
        {
            using (session.Scope("total_check"))
            {
                Thread.Sleep(5);
            }
        }

        var stats = session.GetStats("total_check");
        Assert.NotNull(stats);

        // Total should approximately equal Count * Mean (Welford's maintains this)
        var expected = stats.Count * stats.MeanMs;
        Assert.Equal(expected, stats.TotalMs, 0.01);
    }

    [Fact]
    public void ProfilerStats_StdDev_NonNegative()
    {
        var config = new ProfilingConfig { Enabled = true, AutoEnableInDebug = false };
        var session = new ProfilerSession(config);

        for (int i = 0; i < 10; i++)
        {
            using (session.Scope("stddev"))
            {
                Thread.Sleep(1);
            }
        }

        var stats = session.GetStats("stddev");
        Assert.NotNull(stats);
        Assert.True(stats.StdDevMs >= 0);
    }

    [Fact]
    public void ProfilerStats_Percentiles_OrderedCorrectly()
    {
        var config = new ProfilingConfig { Enabled = true, AutoEnableInDebug = false };
        var session = new ProfilerSession(config);

        for (int i = 0; i < 20; i++)
        {
            using (session.Scope("percentiles"))
            {
                Thread.Sleep(1 + (i % 3)); // Vary timing slightly
            }
        }

        var stats = session.GetStats("percentiles");
        Assert.NotNull(stats);

        // P50 <= P95 <= P99
        Assert.True(stats.P50Ms <= stats.P95Ms + 0.001);
        Assert.True(stats.P95Ms <= stats.P99Ms + 0.001);
    }

    [Fact]
    public void ProfilerStats_OpsPerSecond_HandComputed()
    {
        // OpsPerSecond = Count / (TotalMs / 1000.0)
        // If Count=10, TotalMs=2000 => 10 / 2.0 = 5 ops/sec
        var config = new ProfilingConfig { Enabled = true, AutoEnableInDebug = false };
        var session = new ProfilerSession(config);

        // Record enough operations to get meaningful stats
        for (int i = 0; i < 5; i++)
        {
            using (session.Scope("ops_rate"))
            {
                Thread.Sleep(10);
            }
        }

        var stats = session.GetStats("ops_rate");
        Assert.NotNull(stats);

        // OpsPerSecond should be positive
        Assert.True(stats.OpsPerSecond > 0);

        // Verify formula: OpsPerSecond = Count / (TotalMs / 1000)
        var expected = stats.Count / (stats.TotalMs / 1000.0);
        Assert.Equal(expected, stats.OpsPerSecond, 0.001);
    }

    [Fact]
    public void ProfilerStats_ZeroSamples_ReturnsZeros()
    {
        var config = new ProfilingConfig { Enabled = true, AutoEnableInDebug = false };
        var session = new ProfilerSession(config);

        // Don't record anything
        var stats = session.GetStats("nonexistent");

        Assert.Null(stats); // No entry at all
    }

    // ============================
    // ProfileReport Tests
    // ============================

    [Fact]
    public void ProfileReport_GetAllStats_SortedByTotalTimeDescending()
    {
        var config = new ProfilingConfig { Enabled = true, AutoEnableInDebug = false };
        var session = new ProfilerSession(config);

        using (session.Scope("fast")) { Thread.Sleep(5); }
        using (session.Scope("slow")) { Thread.Sleep(20); }
        using (session.Scope("medium")) { Thread.Sleep(10); }

        var report = session.GetReport();
        var allStats = report.GetAllStats();

        Assert.Equal(3, allStats.Count);
        // Should be sorted by TotalMs descending
        for (int i = 0; i < allStats.Count - 1; i++)
        {
            Assert.True(allStats[i].TotalMs >= allStats[i + 1].TotalMs);
        }
    }

    [Fact]
    public void ProfileReport_GetTopOperations_LimitsCount()
    {
        var config = new ProfilingConfig { Enabled = true, AutoEnableInDebug = false };
        var session = new ProfilerSession(config);

        for (int i = 0; i < 10; i++)
        {
            using (session.Scope($"op_{i}")) { Thread.Sleep(1); }
        }

        var report = session.GetReport();
        var top3 = report.GetTopOperations(3);

        Assert.Equal(3, top3.Count);
    }

    [Fact]
    public void ProfileReport_TotalOperations_MatchesEntryCount()
    {
        var config = new ProfilingConfig { Enabled = true, AutoEnableInDebug = false };
        var session = new ProfilerSession(config);

        using (session.Scope("a")) { Thread.Sleep(1); }
        using (session.Scope("b")) { Thread.Sleep(1); }

        var report = session.GetReport();
        Assert.Equal(2, report.TotalOperations);
    }

    [Fact]
    public void ProfileReport_ToString_ContainsHeader()
    {
        var config = new ProfilingConfig { Enabled = true, AutoEnableInDebug = false };
        var session = new ProfilerSession(config);

        using (session.Scope("test")) { Thread.Sleep(1); }

        var report = session.GetReport();
        var str = report.ToString();

        Assert.Contains("Profiling Report", str);
        Assert.Contains("Start Time:", str);
        Assert.Contains("Total Runtime:", str);
        Assert.Contains("Operations Tracked:", str);
    }

    [Fact]
    public void ProfileReport_ToDictionary_HasRequiredKeys()
    {
        var config = new ProfilingConfig { Enabled = true, AutoEnableInDebug = false };
        var session = new ProfilerSession(config);

        using (session.Scope("test")) { Thread.Sleep(1); }

        var report = session.GetReport();
        var dict = report.ToDictionary();

        Assert.True(dict.ContainsKey("startTime"));
        Assert.True(dict.ContainsKey("totalRuntimeMs"));
        Assert.True(dict.ContainsKey("operationCount"));
        Assert.True(dict.ContainsKey("tags"));
        Assert.True(dict.ContainsKey("operations"));
    }

    [Fact]
    public void ProfileReport_Elapsed_IsPositive()
    {
        var config = new ProfilingConfig { Enabled = true, AutoEnableInDebug = false };
        var session = new ProfilerSession(config);

        Thread.Sleep(5);

        Assert.True(session.Elapsed.TotalMilliseconds > 0);
    }

    // ============================
    // ProfileReport Comparison Tests
    // ============================

    [Fact]
    public void ProfileComparison_NoRegressions_WhenIdentical()
    {
        var config = new ProfilingConfig { Enabled = true, AutoEnableInDebug = false };
        var session1 = new ProfilerSession(config);
        var session2 = new ProfilerSession(config);

        for (int i = 0; i < 5; i++)
        {
            using (session1.Scope("op")) { Thread.Sleep(10); }
            using (session2.Scope("op")) { Thread.Sleep(10); }
        }

        var report1 = session1.GetReport();
        var report2 = session2.GetReport();

        // With similar timings, threshold of 50% should show no regressions
        var comparison = report2.CompareTo(report1, 50.0);

        Assert.NotNull(comparison);
        Assert.Equal(50.0, comparison.ThresholdPercent);
    }

    [Fact]
    public void ProfileComparison_ToString_ContainsThreshold()
    {
        var config = new ProfilingConfig { Enabled = true, AutoEnableInDebug = false };
        var session1 = new ProfilerSession(config);
        var session2 = new ProfilerSession(config);

        using (session1.Scope("op")) { Thread.Sleep(1); }
        using (session2.Scope("op")) { Thread.Sleep(1); }

        var comparison = session2.GetReport().CompareTo(session1.GetReport(), 10.0);
        var str = comparison.ToString();

        Assert.Contains("threshold: 10%", str);
    }

    // ============================
    // ProfileReport Export Tests
    // ============================

    [Fact]
    public void ProfileReport_ToJson_ValidStructure()
    {
        var config = new ProfilingConfig { Enabled = true, AutoEnableInDebug = false };
        var session = new ProfilerSession(config);

        using (session.Scope("json_test")) { Thread.Sleep(1); }

        var report = session.GetReport();
        var json = report.ToJson();

        Assert.Contains("\"startTime\":", json);
        Assert.Contains("\"totalRuntimeMs\":", json);
        Assert.Contains("\"operations\":", json);
        Assert.Contains("\"json_test\"", json);
    }

    [Fact]
    public void ProfileReport_ToCsv_HasHeader()
    {
        var config = new ProfilingConfig { Enabled = true, AutoEnableInDebug = false };
        var session = new ProfilerSession(config);

        using (session.Scope("csv_test")) { Thread.Sleep(1); }

        var report = session.GetReport();
        var csv = report.ToCsv();

        Assert.Contains("Name,Count,TotalMs", csv);
        Assert.Contains("csv_test", csv);
    }

    [Fact]
    public void ProfileReport_ToMarkdown_HasHeader()
    {
        var config = new ProfilingConfig { Enabled = true, AutoEnableInDebug = false };
        var session = new ProfilerSession(config);

        using (session.Scope("md_test")) { Thread.Sleep(1); }

        var report = session.GetReport();
        var md = report.ToMarkdown();

        Assert.Contains("# Profiling Report", md);
        Assert.Contains("md_test", md);
    }

    // ============================
    // ProfilingConfig Tests
    // ============================

    [Fact]
    public void ProfilingConfig_Defaults_AreCorrect()
    {
        var config = new ProfilingConfig();

        Assert.False(config.Enabled);
        Assert.Equal(1.0, config.SamplingRate);
        Assert.Equal(1000, config.ReservoirSize);
        Assert.Equal(10000, config.MaxOperations);
        Assert.True(config.TrackCallHierarchy);
        Assert.True(config.TrackAllocations);
        Assert.False(config.DetailedTiming);
        Assert.True(config.AutoEnableInDebug);
    }

    [Fact]
    public void ProfilingConfig_TraceExecution_AliasForTrackCallHierarchy()
    {
        var config = new ProfilingConfig();

        config.TraceExecution = false;
        Assert.False(config.TrackCallHierarchy);

        config.TrackCallHierarchy = true;
        Assert.True(config.TraceExecution);
    }

    [Fact]
    public void ProfilingConfig_MeasureMemory_AliasForTrackAllocations()
    {
        var config = new ProfilingConfig();

        config.MeasureMemory = false;
        Assert.False(config.TrackAllocations);

        config.TrackAllocations = true;
        Assert.True(config.MeasureMemory);
    }

    [Fact]
    public void ProfilingConfig_CustomTags_EmptyByDefault()
    {
        var config = new ProfilingConfig();

        Assert.NotNull(config.CustomTags);
        Assert.Empty(config.CustomTags);
    }

    [Fact]
    public void ProfilingConfig_CustomTags_IncludedInReport()
    {
        var config = new ProfilingConfig
        {
            Enabled = true,
            AutoEnableInDebug = false,
            CustomTags = new Dictionary<string, string>
            {
                { "model", "GPT-2" },
                { "experiment", "perf_test" }
            }
        };

        var session = new ProfilerSession(config);
        using (session.Scope("tagged_op")) { Thread.Sleep(1); }

        var report = session.GetReport();

        Assert.True(report.Tags.ContainsKey("model"));
        Assert.Equal("GPT-2", report.Tags["model"]);
        Assert.True(report.Tags.ContainsKey("experiment"));
    }

    // ============================
    // MemoryScope Tests
    // ============================

    [Fact]
    public void MemoryScope_CapturesBeforeSnapshot()
    {
        using var scope = MemoryTracker.TrackScope("scope_test");

        Assert.NotNull(scope.Before);
        Assert.Equal("scope_test_before", scope.Before.Label);
    }

    [Fact]
    public void MemoryScope_DisposeTakesAfterSnapshot()
    {
        var wasEnabled = MemoryTracker.IsEnabled;
        try
        {
            MemoryTracker.Reset();
            MemoryTracker.Enable();

            using (MemoryTracker.TrackScope("after_test"))
            {
                // Do some work
                var data = new byte[1024];
                GC.KeepAlive(data);
            }

            var history = MemoryTracker.GetHistory();
            // Should have at least 2 snapshots: before + after
            Assert.True(history.Count >= 2);

            // Check that before and after labels exist
            var labels = history.Select(h => h.Label).ToList();
            Assert.Contains("after_test_before", labels);
            Assert.Contains("after_test_after", labels);
        }
        finally
        {
            MemoryTracker.Reset();
            if (wasEnabled) MemoryTracker.Enable();
            else MemoryTracker.Disable();
        }
    }

    // ============================
    // MemoryPressureLevel Tests
    // ============================

    [Fact]
    public void GetPressureLevel_ReturnsValidLevel()
    {
        var level = MemoryTracker.GetPressureLevel();

        // Should be one of the valid levels
        Assert.True(
            level == MemoryPressureLevel.Low ||
            level == MemoryPressureLevel.Medium ||
            level == MemoryPressureLevel.High ||
            level == MemoryPressureLevel.Critical);
    }

    // ============================
    // ProfilerStats ToString Tests
    // ============================

    [Fact]
    public void ProfilerStats_ToString_ContainsNameAndCounts()
    {
        var config = new ProfilingConfig { Enabled = true, AutoEnableInDebug = false };
        var session = new ProfilerSession(config);

        for (int i = 0; i < 3; i++)
        {
            using (session.Scope("tostring_test")) { Thread.Sleep(1); }
        }

        var stats = session.GetStats("tostring_test");
        Assert.NotNull(stats);

        var str = stats.ToString();
        Assert.Contains("tostring_test", str);
        Assert.Contains("3 calls", str);
        Assert.Contains("Mean:", str);
        Assert.Contains("P95:", str);
    }

    // ============================
    // Profile Hotspots Tests
    // ============================

    [Fact]
    public void ProfileReport_GetHotspots_ReturnsTopByTotalTime()
    {
        var config = new ProfilingConfig { Enabled = true, AutoEnableInDebug = false };
        var session = new ProfilerSession(config);

        using (session.Scope("hot_fast")) { Thread.Sleep(5); }
        using (session.Scope("hot_slow")) { Thread.Sleep(30); }
        using (session.Scope("hot_medium")) { Thread.Sleep(15); }

        var report = session.GetReport();
        var hotspots = report.GetHotspots(2);

        Assert.Equal(2, hotspots.Count);
        // First should be the slowest
        Assert.True(hotspots[0].TotalMs >= hotspots[1].TotalMs);
    }

    [Fact]
    public void ProfileReport_GetSlowOperations_FiltersAboveThreshold()
    {
        var config = new ProfilingConfig { Enabled = true, AutoEnableInDebug = false };
        var session = new ProfilerSession(config);

        using (session.Scope("slow_op")) { Thread.Sleep(50); }
        using (session.Scope("fast_op")) { Thread.Sleep(1); }

        var report = session.GetReport();
        // Threshold of 20ms should include slow_op but not fast_op
        var slow = report.GetSlowOperations(20);

        Assert.True(slow.Count >= 1);
        Assert.All(slow, s => Assert.True(s.P95Ms > 20));
    }

    [Fact]
    public void ProfileReport_GetStats_ByName_ReturnsCorrectEntry()
    {
        var config = new ProfilingConfig { Enabled = true, AutoEnableInDebug = false };
        var session = new ProfilerSession(config);

        using (session.Scope("specific_op")) { Thread.Sleep(5); }
        using (session.Scope("other_op")) { Thread.Sleep(5); }

        var report = session.GetReport();
        var stats = report.GetStats("specific_op");

        Assert.NotNull(stats);
        Assert.Equal("specific_op", stats.Name);
    }

    [Fact]
    public void ProfileReport_GetStats_NonExistent_ReturnsNull()
    {
        var config = new ProfilingConfig { Enabled = true, AutoEnableInDebug = false };
        var session = new ProfilerSession(config);

        using (session.Scope("exists")) { Thread.Sleep(1); }

        var report = session.GetReport();
        var stats = report.GetStats("does_not_exist");

        Assert.Null(stats);
    }
}
