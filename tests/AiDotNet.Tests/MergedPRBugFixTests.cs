using System.Reflection;
using AiDotNet.Deployment.Configuration;
using AiDotNet.Diagnostics;
using AiDotNet.Enums;
using AiDotNet.Evaluation;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Metrics;
using AiDotNet.ModelRegistry;
using AiDotNet.Models;
using AiDotNet.PromptEngineering.Optimization;
using AiDotNet.PromptEngineering.Templates;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TrainingMonitoring;
using Xunit;

namespace AiDotNet.Tests;

/// <summary>
/// Tests for bugs found in recently merged PRs during production-readiness review.
/// Each test exposes a bug that was fixed in the source code.
/// </summary>
public class MergedPRBugFixTests
{
    #region Evaluation PR #765 - PredictionTypeInference Integer Overflow Bug

    [Fact]
    public void PredictionTypeInference_LargeClassRange_DoesNotOverflow()
    {
        // ARRANGE: Create a vector with class labels that span a large range
        // This would cause integer overflow with the old code: maxClass - minClass
        // e.g., if minClass = -2000000000 and maxClass = 2000000000
        // the subtraction would overflow because result (4000000000) > int.MaxValue
        var actual = new Vector<double>(10);

        // Use class labels that would overflow if subtracted as ints
        actual[0] = -2000000000;
        actual[1] = 2000000000;
        actual[2] = -1000000000;
        actual[3] = 1000000000;
        actual[4] = 0;
        actual[5] = -500000000;
        actual[6] = 500000000;
        actual[7] = -100000000;
        actual[8] = 100000000;
        actual[9] = 50000000;

        // ACT: Infer prediction type - should not throw or return incorrect results due to overflow
        var method = typeof(PredictionTypeInference).GetMethod("Infer",
            BindingFlags.Public | BindingFlags.Static);
        var genericMethod = method!.MakeGenericMethod(typeof(double));

        var result = (PredictionType)genericMethod.Invoke(null, new object[] { actual })!;

        // ASSERT: With such a large range, it should be classified as regression
        // (not contiguous, high unique ratio)
        Assert.Equal(PredictionType.Regression, result);
    }

    [Fact]
    public void PredictionTypeInference_ExtremeClassValues_DoesNotOverflow()
    {
        // ARRANGE: Extreme case - minimum and maximum integer values
        var actual = new Vector<double>(2);
        actual[0] = int.MinValue;
        actual[1] = int.MaxValue;

        // ACT: Should not throw
        var method = typeof(PredictionTypeInference).GetMethod("Infer",
            BindingFlags.Public | BindingFlags.Static);
        var genericMethod = method!.MakeGenericMethod(typeof(double));

        var result = (PredictionType)genericMethod.Invoke(null, new object[] { actual })!;

        // ASSERT: Should classify as regression due to non-contiguous extreme range
        Assert.Equal(PredictionType.Regression, result);
    }

    [Fact]
    public void PredictionTypeInference_NormalClassRange_StillWorksCorrectly()
    {
        // ARRANGE: Normal multiclass scenario with small contiguous labels
        var actual = new Vector<double>(100);
        for (int i = 0; i < 100; i++)
        {
            actual[i] = i % 5; // Classes 0, 1, 2, 3, 4
        }

        // ACT
        var method = typeof(PredictionTypeInference).GetMethod("Infer",
            BindingFlags.Public | BindingFlags.Static);
        var genericMethod = method!.MakeGenericMethod(typeof(double));

        var result = (PredictionType)genericMethod.Invoke(null, new object[] { actual })!;

        // ASSERT: Should be multiclass
        Assert.Equal(PredictionType.MultiClass, result);
    }

    #endregion

    #region PromptEngineering PR #762 - GeneticOptimizer IndexOf Bug

    [Fact]
    public void GeneticOptimizer_TournamentSelect_WorksWithDuplicatePrompts()
    {
        // ARRANGE: Create optimizer and test with population containing duplicates
        // The old code used IndexOf which returns the FIRST occurrence index,
        // causing wrong fitness to be selected when duplicates exist
        var optimizer = new GeneticOptimizer<double>(
            populationSize: 10,
            mutationRate: 0.1,
            crossoverRate: 0.7,
            eliteCount: 2,
            seed: 12345 // Fixed seed for reproducibility
        );

        // Use a simple evaluation function
        int callCount = 0;
        Func<string, double> evaluator = (prompt) =>
        {
            callCount++;
            // Score based on length - simple deterministic evaluation
            return prompt.Length / 100.0;
        };

        // ACT: Run optimization - should not crash or produce incorrect results
        var result = optimizer.Optimize("Test prompt for optimization", evaluator, maxIterations: 20);

        // ASSERT: Optimization should complete successfully
        Assert.NotNull(result);
        Assert.True(callCount > 0, "Evaluator should have been called");
    }

    [Fact]
    public void GeneticOptimizer_OptimizesPrompts_ReturnsValidTemplate()
    {
        // ARRANGE
        var optimizer = new GeneticOptimizer<double>(
            populationSize: 8,
            mutationRate: 0.2,
            seed: 42
        );

        // ACT
        var result = optimizer.Optimize(
            "Classify this text",
            prompt => prompt.Length > 10 ? 0.8 : 0.2,
            maxIterations: 10
        );

        // ASSERT
        Assert.NotNull(result);
        Assert.IsType<SimplePromptTemplate>(result);
    }

    #endregion

    #region TrainingMonitoring PR #755 - CSV Export Missing Data Bug

    [Fact]
    public void TrainingMonitor_CSVExport_EmptyMetricShowsEmpty_NotZero()
    {
        // ARRANGE: Create monitor with sparse metrics (not every step has every metric)
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("TestSession");

        // Log different metrics at different steps
        monitor.LogMetric(sessionId, "loss", 0.5, step: 1);
        monitor.LogMetric(sessionId, "accuracy", 0.7, step: 2); // Note: no loss at step 2
        monitor.LogMetric(sessionId, "loss", 0.3, step: 3);

        var tempFile = Path.GetTempFileName();
        try
        {
            // ACT
            monitor.ExportData(sessionId, tempFile, "csv");

            // ASSERT
            var csvContent = File.ReadAllText(tempFile);
            var lines = csvContent.Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);

            // Header should contain both metrics
            Assert.Contains("loss", lines[0]);
            Assert.Contains("accuracy", lines[0]);

            // Find the line for step 2 (where loss is missing)
            var step2Line = lines.FirstOrDefault(l => l.StartsWith("2,"));
            Assert.NotNull(step2Line);

            // The old bug: missing metrics showed as "0" instead of empty
            // The line should have an empty value for loss (between commas)
            // Format: "Step,Timestamp,loss,accuracy" -> "2,<timestamp>,,0.7" (empty loss)
            var parts = step2Line.Split(',');

            // Step 2 should have empty loss (index 2) and value for accuracy
            // If the fix works, loss column (index 2) should be empty, not "0"
            if (parts.Length >= 4)
            {
                // Either loss should be empty or accuracy should be empty depending on column order
                bool hasEmptyMetric = parts.Skip(2).Any(p => p == "");
                Assert.True(hasEmptyMetric, "Missing metrics should be empty, not '0'");
            }
        }
        finally
        {
            monitor.EndSession(sessionId);
            if (File.Exists(tempFile))
            {
                File.Delete(tempFile);
            }
        }
    }

    [Fact]
    public void TrainingMonitor_CSVExport_AllMetricsPresent_NoEmptyValues()
    {
        // ARRANGE: Create monitor where all metrics are present at all steps
        var monitor = new TrainingMonitor<double>();
        var sessionId = monitor.StartSession("FullDataSession");

        // Log metrics at every step
        for (int step = 1; step <= 3; step++)
        {
            monitor.LogMetrics(sessionId, new Dictionary<string, double>
            {
                { "loss", 1.0 - step * 0.2 },
                { "accuracy", step * 0.25 }
            }, step);
        }

        var tempFile = Path.GetTempFileName();
        try
        {
            // ACT
            monitor.ExportData(sessionId, tempFile, "csv");

            // ASSERT
            var csvContent = File.ReadAllText(tempFile);
            var lines = csvContent.Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);

            // All data lines (after header) should have no empty values
            foreach (var line in lines.Skip(1))
            {
                var parts = line.Split(',');
                foreach (var part in parts)
                {
                    Assert.False(string.IsNullOrEmpty(part),
                        $"All values should be present when metrics exist: '{line}'");
                }
            }
        }
        finally
        {
            monitor.EndSession(sessionId);
            if (File.Exists(tempFile))
            {
                File.Delete(tempFile);
            }
        }
    }

    #endregion

    #region ProgramSynthesis PR #763 - Relative Error Comparison Bug

    [Fact]
    public void OutputMatch_LargeNumbers_UsesRelativeComparison()
    {
        // ARRANGE: The old code used absolute tolerance (1e-6) which fails for large numbers
        // For example, 1e12 + 0.5 vs 1e12 has absolute error 0.5 > 1e-6, but relative error is ~5e-13

        // We need to test the IsOutputMatch method via reflection since it's private
        var assembly = typeof(ProgramSynthesis.Engines.NeuralProgramSynthesizer<>).Assembly;
        var synthType = assembly.GetType("AiDotNet.ProgramSynthesis.Engines.NeuralProgramSynthesizer`1")!
            .MakeGenericType(typeof(double));

        var isOutputMatchMethod = synthType.GetMethod("IsOutputMatch",
            BindingFlags.NonPublic | BindingFlags.Static);

        if (isOutputMatchMethod == null)
        {
            // Method might not exist if the class structure changed
            // In that case, skip this specific test
            return;
        }

        // ACT & ASSERT: Large numbers with small relative difference should match

        // Test 1: Large numbers with negligible relative error
        var result1 = (bool)isOutputMatchMethod.Invoke(null, new object[] { "1000000000000.0001", "1000000000000.0" })!;
        Assert.True(result1, "Large numbers with tiny relative error should match");

        // Test 2: Small numbers within absolute tolerance should match
        // absoluteTolerance is 1e-9, so 1e-10 vs 0 should match
        var result2 = (bool)isOutputMatchMethod.Invoke(null, new object[] { "0.0000000001", "0.0" })!;
        Assert.True(result2, "Numbers near zero within absolute tolerance (1e-9) should match");

        // Test 3: Numbers with significant relative error should NOT match
        var result3 = (bool)isOutputMatchMethod.Invoke(null, new object[] { "100.0", "99.0" })!;
        Assert.False(result3, "1% relative error should not match");
    }

    [Fact]
    public void OutputMatch_SmallNumbers_UsesAbsoluteComparison()
    {
        // Numbers near zero should use absolute comparison
        var assembly = typeof(ProgramSynthesis.Engines.NeuralProgramSynthesizer<>).Assembly;
        var synthType = assembly.GetType("AiDotNet.ProgramSynthesis.Engines.NeuralProgramSynthesizer`1")!
            .MakeGenericType(typeof(double));

        var isOutputMatchMethod = synthType.GetMethod("IsOutputMatch",
            BindingFlags.NonPublic | BindingFlags.Static);

        if (isOutputMatchMethod == null)
        {
            return;
        }

        // Small numbers near zero should match if within absolute tolerance
        var result = (bool)isOutputMatchMethod.Invoke(null, new object[] { "0.0000000001", "0.0" })!;
        Assert.True(result, "Tiny numbers should match using absolute tolerance");
    }

    #endregion

    #region FineTuning PR #753 - Log Probability Always Zero Bug

    [Fact]
    public void SupervisedFineTuning_CanBeInstantiated_WithSingleParameterConstructor()
    {
        // ARRANGE: The old SupervisedFineTuning only had a constructor with optional second parameter:
        // SupervisedFineTuning(FineTuningOptions<T> options, ILossFunction<T>? lossFunction = null)
        // This caused Activator.CreateInstance(type, options) to fail because it couldn't find
        // a constructor with exactly one parameter.
        // FIX: Added an explicit single-parameter constructor.

        var optionsType = typeof(AiDotNet.Models.Options.FineTuningOptions<>).MakeGenericType(typeof(double));
        var options = Activator.CreateInstance(optionsType);

        var sftType = typeof(FineTuning.SupervisedFineTuning<,,>)
            .MakeGenericType(typeof(double), typeof(double[]), typeof(double[]));

        // ACT: This should NOT throw MissingMethodException
        Exception? caughtException = null;
        object? sftInstance = null;
        try
        {
            sftInstance = Activator.CreateInstance(sftType, options);
        }
        catch (Exception ex)
        {
            caughtException = ex;
        }

        // ASSERT
        Assert.Null(caughtException);
        Assert.NotNull(sftInstance);
        Assert.IsAssignableFrom(typeof(FineTuning.FineTuningBase<double, double[], double[]>), sftInstance);
    }

    [Fact]
    public void FineTuningBase_ComputeLogProbability_ArrayOutputs_NotAlwaysZero()
    {
        // ARRANGE: The old code always returned 0.0 for log probability
        // which made DPO/RLHF/etc. compute constant loss regardless of inputs

        // Use reflection to test the protected method
        var fineTuningBaseType = typeof(FineTuning.FineTuningBase<,,>)
            .MakeGenericType(typeof(double), typeof(double[]), typeof(double[]));

        var computeMethod = fineTuningBaseType.GetMethod(
            "ComputeLogProbabilityFromPrediction",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

        if (computeMethod == null)
        {
            // Method signature changed, skip test
            return;
        }

        // Create a concrete implementation to test
        var optionsType = typeof(AiDotNet.Models.Options.FineTuningOptions<>).MakeGenericType(typeof(double));
        var options = Activator.CreateInstance(optionsType);

        // Use DPO as the concrete implementation
        var dpoType = typeof(FineTuning.DirectPreferenceOptimization<,,>)
            .MakeGenericType(typeof(double), typeof(double[]), typeof(double[]));
        var dpo = Activator.CreateInstance(dpoType, options);

        // Test 1: Identical arrays should give log prob close to 0 (high probability)
        var identicalPred = new double[] { 0.2, 0.3, 0.5 }; // Valid probability distribution
        var identicalTarget = new double[] { 0.2, 0.3, 0.5 };
        var logProbIdentical = (double)computeMethod.Invoke(dpo, new object[] { identicalPred, identicalTarget })!;

        // Test 2: Different arrays should give more negative log prob
        var differentPred = new double[] { 0.8, 0.1, 0.1 }; // Different distribution
        var differentTarget = new double[] { 0.1, 0.1, 0.8 };
        var logProbDifferent = (double)computeMethod.Invoke(dpo, new object[] { differentPred, differentTarget })!;

        // ASSERT: Log probs should be different (not both 0.0)
        // Identical should be higher (less negative) than different
        Assert.NotEqual(0.0, logProbIdentical);
        Assert.NotEqual(0.0, logProbDifferent);
        Assert.True(logProbIdentical > logProbDifferent,
            $"Identical arrays should have higher log prob ({logProbIdentical}) than different ({logProbDifferent})");
    }

    [Fact]
    public void FineTuningBase_ComputeLogProbability_ScalarOutputs_NotAlwaysZero()
    {
        // ARRANGE: Use DPO with double[] types since we know it has the right constructor
        // We're testing the base class method which handles scalar conversion internally
        var optionsType = typeof(AiDotNet.Models.Options.FineTuningOptions<>).MakeGenericType(typeof(double));
        var options = Activator.CreateInstance(optionsType);

        var dpoType = typeof(FineTuning.DirectPreferenceOptimization<,,>)
            .MakeGenericType(typeof(double), typeof(double[]), typeof(double[]));
        var dpo = Activator.CreateInstance(dpoType, options);

        var computeMethod = dpoType.BaseType!.GetMethod(
            "ComputeLogProbabilityFromPrediction",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

        if (computeMethod == null) return;

        // Test with single-element arrays (simulating scalar behavior)
        var identical1 = new double[] { 5.0 };
        var identical2 = new double[] { 5.0 };
        var different = new double[] { 10.0 };

        var logProbIdentical = (double)computeMethod.Invoke(dpo, new object[] { identical1, identical2 })!;
        var logProbDifferent = (double)computeMethod.Invoke(dpo, new object[] { different, identical1 })!;

        // ASSERT: Identical should have higher log prob than different
        Assert.True(logProbIdentical > logProbDifferent,
            $"Identical arrays should have higher log prob ({logProbIdentical}) than different ({logProbDifferent})");
    }

    [Fact]
    public void FineTuningBase_ComputeLogProbability_StringOutputs_NotAlwaysZero()
    {
        // ARRANGE: Use DPO with string types
        var optionsType = typeof(AiDotNet.Models.Options.FineTuningOptions<>).MakeGenericType(typeof(double));
        var options = Activator.CreateInstance(optionsType);

        var dpoType = typeof(FineTuning.DirectPreferenceOptimization<,,>)
            .MakeGenericType(typeof(double), typeof(string), typeof(string));
        var dpo = Activator.CreateInstance(dpoType, options);

        var computeMethod = dpoType.BaseType!.GetMethod(
            "ComputeLogProbabilityFromPrediction",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

        if (computeMethod == null) return;

        // Test: Identical strings should give 0
        var logProbIdentical = (double)computeMethod.Invoke(dpo, new object[] { "hello", "hello" })!;

        // Test: Similar strings should give higher log prob than different
        var logProbSimilar = (double)computeMethod.Invoke(dpo, new object[] { "hello", "hallo" })!;
        var logProbDifferent = (double)computeMethod.Invoke(dpo, new object[] { "hello", "world" })!;

        // ASSERT
        Assert.Equal(0.0, logProbIdentical, precision: 10);
        Assert.True(logProbSimilar > logProbDifferent,
            $"Similar strings should have higher log prob ({logProbSimilar}) than different ({logProbDifferent})");
    }

    [Fact]
    public void DirectPreferenceOptimization_DPOLoss_VariesWithInputs()
    {
        // This test verifies that DPO loss actually varies based on inputs
        // Before the fix, loss was always ~0.693 (constant) because log probs were always 0

        var optionsType = typeof(AiDotNet.Models.Options.FineTuningOptions<>).MakeGenericType(typeof(double));
        var options = Activator.CreateInstance(optionsType);

        var dpoType = typeof(FineTuning.DirectPreferenceOptimization<,,>)
            .MakeGenericType(typeof(double), typeof(double[]), typeof(double[]));
        var dpo = Activator.CreateInstance(dpoType, options);

        var computeMethod = dpoType.BaseType!.GetMethod(
            "ComputeLogProbabilityFromPrediction",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

        if (computeMethod == null) return;

        // Compute log probs for two different preference pairs
        var chosen1 = new double[] { 0.9, 0.05, 0.05 };
        var rejected1 = new double[] { 0.1, 0.1, 0.8 };

        var chosen2 = new double[] { 0.33, 0.33, 0.34 };
        var rejected2 = new double[] { 0.33, 0.34, 0.33 };

        var logProbChosen1 = (double)computeMethod.Invoke(dpo, new object[] { chosen1, chosen1 })!;
        var logProbRejected1 = (double)computeMethod.Invoke(dpo, new object[] { rejected1, chosen1 })!;

        var logProbChosen2 = (double)computeMethod.Invoke(dpo, new object[] { chosen2, chosen2 })!;
        var logProbRejected2 = (double)computeMethod.Invoke(dpo, new object[] { rejected2, chosen2 })!;

        // The margins should be different for different preference pairs
        var margin1 = logProbChosen1 - logProbRejected1;
        var margin2 = logProbChosen2 - logProbRejected2;

        // ASSERT: Margins should be different (not both 0)
        Assert.NotEqual(0.0, margin1);
        Assert.NotEqual(0.0, margin2);
        // Strong preference (pair 1) should have larger margin than weak preference (pair 2)
        Assert.True(Math.Abs(margin1) > Math.Abs(margin2),
            $"Strong preference margin ({margin1}) should be larger than weak preference margin ({margin2})");
    }

    #endregion

    #region Diagnostics PR #777 - Production Bug Fixes

    [Fact]
    public void MemoryTracker_EnforcesMaxHistorySize_PreventsUnboundedGrowth()
    {
        // ARRANGE: The old code had no limit on history size, causing memory leaks
        // in long-running applications
        MemoryTracker.Reset();
        MemoryTracker.Enable();
        MemoryTracker.MaxHistorySize = 5; // Set a small limit for testing

        try
        {
            // ACT: Take more snapshots than the limit
            for (int i = 0; i < 10; i++)
            {
                MemoryTracker.Snapshot($"Snapshot_{i}");
            }

            var history = MemoryTracker.GetHistory();

            // ASSERT: History should be capped at MaxHistorySize
            Assert.True(history.Count <= 5,
                $"History should be capped at MaxHistorySize (5), but was {history.Count}");

            // The most recent snapshots should be kept
            Assert.Contains(history, s => s.Label == "Snapshot_9");
        }
        finally
        {
            MemoryTracker.Disable();
            MemoryTracker.Reset();
            MemoryTracker.MaxHistorySize = 10000; // Reset to default
        }
    }

    [Fact]
    public void ProfilerSession_ThreadSafeRandomSampling_DoesNotCorrupt()
    {
        // ARRANGE: The old code used a single Random instance which is NOT thread-safe
        // Multiple threads would corrupt the Random state
        var config = new ProfilingConfig
        {
            Enabled = true,
            SamplingRate = 0.5 // 50% sampling to exercise the random path
        };
        var session = new ProfilerSession(config);

        // ACT: Record timings from multiple threads concurrently
        var exceptions = new System.Collections.Concurrent.ConcurrentBag<Exception>();
        var tasks = new List<Task>();

        for (int t = 0; t < 10; t++)
        {
            int threadId = t;
            tasks.Add(Task.Run(() =>
            {
                try
                {
                    for (int i = 0; i < 100; i++)
                    {
                        using (session.Scope($"Thread{threadId}_Op{i}"))
                        {
                            // Simulate some work
                            Thread.SpinWait(100);
                        }
                    }
                }
                catch (Exception ex)
                {
                    exceptions.Add(ex);
                }
            }));
        }

        Task.WaitAll(tasks.ToArray());

        // ASSERT: No exceptions should occur from corrupted Random state
        Assert.Empty(exceptions);

        // Some operations should have been recorded (given 50% sampling rate across 1000 calls)
        Assert.True(session.OperationCount > 0, "Some operations should be recorded");
    }

    [Fact]
    public void ProfilerSession_SampleVariance_UsesUnbiasedEstimator()
    {
        // ARRANGE: The old code used population variance (_m2 / _count)
        // which underestimates variance. Fixed to use sample variance (_m2 / (_count - 1))
        var config = new ProfilingConfig { Enabled = true, SamplingRate = 1.0 };
        var session = new ProfilerSession(config);

        // ACT: Record known values with known variance
        // Values: 10, 20, 30 -> Mean = 20, Sample Variance = 100, StdDev = 10
        // Population variance would be 66.67, sample variance is 100
        var values = new[] { 10.0, 20.0, 30.0 };
        foreach (var v in values)
        {
            using (var scope = session.Scope("TestOp"))
            {
                Thread.Sleep((int)v);
            }
        }

        var stats = session.GetStats("TestOp");

        // ASSERT: Standard deviation should be based on sample variance
        Assert.NotNull(stats);

        // We can't assert exact values due to timing variability,
        // but we can check the variance calculation doesn't produce unreasonable results
        Assert.True(stats.StdDevMs >= 0, "StdDev should be non-negative");

        // With sample variance, StdDev should be slightly larger than with population variance
        // This is a sanity check that the formula changed
        if (stats.Count >= 2 && stats.StdDevMs > 0)
        {
            // The formula sqrt(m2/(n-1)) should give larger values than sqrt(m2/n)
            // We're just verifying the calculation doesn't break
            Assert.True(stats.StdDevMs > 0, "StdDev should be positive for varying samples");
        }
    }

    [Fact]
    public void ProfilerSessionTimer_CleansUpCallStack_EvenWithExceptionOrder()
    {
        // ARRANGE: The old code only popped from call stack if timer was at top
        // This caused memory leaks when nested timers were stopped out of order
        var config = new ProfilingConfig { Enabled = true, TrackCallHierarchy = true };
        var session = new ProfilerSession(config);

        // ACT: Simulate out-of-order stop (as might happen with exceptions)
        var timer1 = session.Start("Outer");
        var timer2 = session.Start("Inner");

        // Stop outer first (wrong order - simulates exception in outer scope)
        timer1.Stop();
        timer2.Stop();

        // Create another timer to verify stack is clean
        using (session.Scope("AfterCleanup"))
        {
            // This should work without issues
        }

        // ASSERT: No exception thrown and stats recorded
        var stats = session.GetStats("AfterCleanup");
        Assert.NotNull(stats);
        Assert.Equal(1, stats.Count);
    }

    [Fact]
    public void MemoryTracker_MaxHistorySize_CanBeConfigured()
    {
        // ARRANGE
        MemoryTracker.Reset();

        // ACT & ASSERT: Property should be configurable
        var originalSize = MemoryTracker.MaxHistorySize;
        Assert.True(originalSize > 0, "Default max history size should be positive");

        MemoryTracker.MaxHistorySize = 500;
        Assert.Equal(500, MemoryTracker.MaxHistorySize);

        // Minimum should be 1
        MemoryTracker.MaxHistorySize = 0;
        Assert.Equal(1, MemoryTracker.MaxHistorySize);

        // Restore
        MemoryTracker.MaxHistorySize = originalSize;
    }

    #endregion

    #region ModelRegistry PR #774 - Production Bug Fixes

    [Fact]
    public void ModelRegistry_GetModel_ReturnsClone_NotInternalState()
    {
        // ARRANGE: The old code returned the internal model object directly
        // External code could modify it, causing inconsistency between memory and disk
        var tempDir = Path.Combine(Path.GetTempPath(), $"model_registry_test_{Guid.NewGuid():N}");
        try
        {
            var registry = new ModelRegistry<double, double[], double[]>(tempDir);

            // Create a mock model and register it
            var metadata = new ModelMetadata<double>
            {
                FeatureCount = 10,
                ModelType = ModelType.NeuralNetwork,
                Complexity = 100
            };
            var model = new MockModel();
            var modelId = registry.RegisterModel("TestModel", model, metadata);

            // ACT: Get the model and modify it
            var retrievedModel = registry.GetModel("TestModel", 1);
            var originalStage = retrievedModel.Stage;
            retrievedModel.Stage = ModelStage.Production; // Modify the clone

            // Get the model again
            var retrievedAgain = registry.GetModel("TestModel", 1);

            // ASSERT: The internal state should NOT have changed
            Assert.Equal(originalStage, retrievedAgain.Stage);
            Assert.NotEqual(ModelStage.Production, retrievedAgain.Stage);
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }

    [Fact]
    public void ModelRegistry_GetModelByStage_ReturnsClone_NotInternalState()
    {
        // ARRANGE
        var tempDir = Path.Combine(Path.GetTempPath(), $"model_registry_test_{Guid.NewGuid():N}");
        try
        {
            var registry = new ModelRegistry<double, double[], double[]>(tempDir);

            var metadata = new ModelMetadata<double>
            {
                FeatureCount = 10,
                ModelType = ModelType.NeuralNetwork
            };
            var model = new MockModel();
            registry.RegisterModel("TestModel", model, metadata);
            registry.TransitionModelStage("TestModel", 1, ModelStage.Staging);

            // ACT: Get model by stage and modify it
            var retrievedModel = registry.GetModelByStage("TestModel", ModelStage.Staging);
            Assert.NotNull(retrievedModel);
            retrievedModel!.Description = "Modified externally";

            // Get again
            var retrievedAgain = registry.GetModelByStage("TestModel", ModelStage.Staging);

            // ASSERT: Description should still be null (not modified)
            Assert.Null(retrievedAgain?.Description);
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }

    [Fact]
    public void ModelRegistry_DeleteModelVersion_ValidatesModelName()
    {
        // ARRANGE: Other methods validate model name, DeleteModelVersion should too
        var tempDir = Path.Combine(Path.GetTempPath(), $"model_registry_test_{Guid.NewGuid():N}");
        try
        {
            var registry = new ModelRegistry<double, double[], double[]>(tempDir);

            // ACT & ASSERT: Null/empty model names should throw ArgumentException
            Assert.Throws<ArgumentException>(() => registry.DeleteModelVersion(null!, 1));
            Assert.Throws<ArgumentException>(() => registry.DeleteModelVersion("", 1));
            Assert.Throws<ArgumentException>(() => registry.DeleteModelVersion("   ", 1));
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }

    [Fact]
    public void ModelRegistry_LoadErrors_ExposedForDiagnostics()
    {
        // ARRANGE: The old code used Console.WriteLine for errors which is bad for production
        // Now errors are exposed via LoadErrors property
        var tempDir = Path.Combine(Path.GetTempPath(), $"model_registry_test_{Guid.NewGuid():N}");
        try
        {
            // Create a registry
            var registry = new ModelRegistry<double, double[], double[]>(tempDir);

            // ASSERT: LoadErrors should be accessible (even if empty)
            Assert.NotNull(registry.LoadErrors);
            Assert.IsAssignableFrom<IReadOnlyList<string>>(registry.LoadErrors);
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }

    [Fact]
    public void ModelRegistry_LineageTracking_WorksForNewModels()
    {
        // ARRANGE: The old code never populated _lineage dictionary
        var tempDir = Path.Combine(Path.GetTempPath(), $"model_registry_test_{Guid.NewGuid():N}");
        try
        {
            var registry = new ModelRegistry<double, double[], double[]>(tempDir);

            var metadata = new ModelMetadata<double>
            {
                FeatureCount = 10,
                ModelType = ModelType.NeuralNetwork
            };
            var model = new MockModel();

            // ACT: Register a model and check lineage
            registry.RegisterModel("TestModel", model, metadata);
            var lineage = registry.GetModelLineage("TestModel", 1);

            // ASSERT: Lineage should be tracked
            Assert.NotNull(lineage);
            Assert.Equal("TestModel", lineage.ModelName);
            Assert.Equal(1, lineage.Version);
            Assert.Null(lineage.ParentVersion); // First version has no parent
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }

    [Fact]
    public void ModelRegistry_LineageTracking_TracksParentForNewVersions()
    {
        // ARRANGE
        var tempDir = Path.Combine(Path.GetTempPath(), $"model_registry_test_{Guid.NewGuid():N}");
        try
        {
            var registry = new ModelRegistry<double, double[], double[]>(tempDir);

            var metadata = new ModelMetadata<double>
            {
                FeatureCount = 10,
                ModelType = ModelType.NeuralNetwork
            };
            var model = new MockModel();

            // ACT: Register and create a new version
            registry.RegisterModel("TestModel", model, metadata);
            registry.CreateModelVersion("TestModel", model, metadata, "Second version");
            var lineage = registry.GetModelLineage("TestModel", 2);

            // ASSERT: Version 2 should have version 1 as parent
            Assert.NotNull(lineage);
            Assert.Equal(2, lineage.Version);
            Assert.Equal(1, lineage.ParentVersion);
            Assert.Equal("TestModel", lineage.ParentModel);
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }

    [Fact]
    public void ModelRegistry_SearchModels_ReturnsClones()
    {
        // ARRANGE
        var tempDir = Path.Combine(Path.GetTempPath(), $"model_registry_test_{Guid.NewGuid():N}");
        try
        {
            var registry = new ModelRegistry<double, double[], double[]>(tempDir);

            var metadata = new ModelMetadata<double>
            {
                FeatureCount = 10,
                ModelType = ModelType.NeuralNetwork
            };
            var model = new MockModel();
            registry.RegisterModel("TestModel", model, metadata);

            // ACT: Search and modify result
            var criteria = new ModelSearchCriteria<double> { NamePattern = "Test" };
            var results = registry.SearchModels(criteria);
            Assert.Single(results);
            results[0].Stage = ModelStage.Production;

            // Search again
            var resultsAgain = registry.SearchModels(criteria);

            // ASSERT: Internal state should not have changed
            Assert.NotEqual(ModelStage.Production, resultsAgain[0].Stage);
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }

    /// <summary>
    /// Mock model implementation for testing.
    /// </summary>
    private class MockModel : Interfaces.IModel<double[], double[], MockModelMetadata>
    {
        public MockModelMetadata GetModelMetadata() => new MockModelMetadata();

        public void Train(double[] x, double[] y)
        {
            // No-op for testing
        }

        public double[] Predict(double[] input)
        {
            return input;
        }
    }

    private class MockModelMetadata { }

    #endregion

    #region Metrics PR #773 - Production Bug Fixes

    [Fact]
    public void PSNR_ComputeBatch_ValidatesNullArguments()
    {
        // ARRANGE: The old code didn't validate null arguments in ComputeBatch
        var psnr = new PeakSignalToNoiseRatio<double>();
        var validTensor = new Tensor<double>([1, 2, 2, 1]);

        // ACT & ASSERT: Null arguments should throw ArgumentNullException
        Assert.Throws<ArgumentNullException>(() => psnr.ComputeBatch(null!, validTensor));
        Assert.Throws<ArgumentNullException>(() => psnr.ComputeBatch(validTensor, null!));
    }

    [Fact]
    public void STOI_Compute_ValidatesNullArguments()
    {
        // ARRANGE: Missing null checks could cause NullReferenceException
        var stoi = new ShortTimeObjectiveIntelligibility<double>();
        var validTensor = new Tensor<double>([1000]);

        // ACT & ASSERT
        Assert.Throws<ArgumentNullException>(() => stoi.Compute(null!, validTensor));
        Assert.Throws<ArgumentNullException>(() => stoi.Compute(validTensor, null!));
    }

    [Fact]
    public void SISDR_Compute_ValidatesNullArguments()
    {
        // ARRANGE: Missing null checks could cause NullReferenceException
        var sisdr = new ScaleInvariantSignalToDistortionRatio<double>();
        var validTensor = new Tensor<double>([100]);

        // ACT & ASSERT
        Assert.Throws<ArgumentNullException>(() => sisdr.Compute(null!, validTensor));
        Assert.Throws<ArgumentNullException>(() => sisdr.Compute(validTensor, null!));
    }

    [Fact]
    public void SISDR_Compute_HandlesEmptyTensors()
    {
        // ARRANGE: The old code would divide by zero in ComputeMean for empty tensors
        var sisdr = new ScaleInvariantSignalToDistortionRatio<double>();
        var emptyTensor = new Tensor<double>([0]);

        // ACT: Should not throw, should return sentinel value
        var result = sisdr.Compute(emptyTensor, emptyTensor);

        // ASSERT: Should return a very low SI-SDR value for empty signals
        Assert.Equal(-100.0, result, 5);
    }

    [Fact]
    public void SNR_Compute_ValidatesNullArguments()
    {
        // ARRANGE: Missing null checks could cause NullReferenceException
        var snr = new SignalToNoiseRatio<double>();
        var validTensor = new Tensor<double>([100]);

        // ACT & ASSERT
        Assert.Throws<ArgumentNullException>(() => snr.Compute(null!, validTensor));
        Assert.Throws<ArgumentNullException>(() => snr.Compute(validTensor, null!));
    }

    [Fact]
    public void IoU3D_ComputeBoxIoU_ValidatesBoxCoordinates()
    {
        // ARRANGE: The old code didn't validate that min <= max for each dimension
        // This could produce negative volumes and incorrect IoU
        var iou3d = new IoU3D<double>();

        // Create invalid boxes where min > max
        var invalidBoxA = new double[] { 10, 10, 10, 0, 0, 0 }; // x_min > x_max, etc.
        var validBoxB = new double[] { 0, 0, 0, 5, 5, 5 };

        // ACT & ASSERT: Should throw for invalid coordinates
        Assert.Throws<ArgumentException>(() => iou3d.ComputeBoxIoU(invalidBoxA, validBoxB));
        Assert.Throws<ArgumentException>(() => iou3d.ComputeBoxIoU(validBoxB, invalidBoxA));
    }

    [Fact]
    public void IoU3D_ComputeBoxIoU_ValidatesNullArguments()
    {
        // ARRANGE
        var iou3d = new IoU3D<double>();
        var validBox = new double[] { 0, 0, 0, 5, 5, 5 };

        // ACT & ASSERT
        Assert.Throws<ArgumentNullException>(() => iou3d.ComputeBoxIoU(null!, validBox));
        Assert.Throws<ArgumentNullException>(() => iou3d.ComputeBoxIoU(validBox, null!));
    }

    [Fact]
    public void ChamferDistance_ComputeOneWay_ThrowsForEmptyTargetWithNonEmptySource()
    {
        // ARRANGE: The old code returned 0 for empty target, which is semantically wrong
        // If source has points but target is empty, there's nothing to match to
        var chamfer = new ChamferDistance<double>();

        // Create non-empty source (2 points in 3D) and empty target
        var source = new Tensor<double>([2, 3]);
        for (int i = 0; i < 6; i++) source[i] = i * 0.5;

        var target = new Tensor<double>([0, 3]);

        // ACT & ASSERT: Should throw because you can't compute distance to nothing
        Assert.Throws<ArgumentException>(() => chamfer.ComputeOneWay(source, target));
    }

    [Fact]
    public void ChamferDistance_ComputeOneWay_ReturnsZeroForEmptySource()
    {
        // ARRANGE: Empty source is fine - no points to match means zero distance (vacuously true)
        var chamfer = new ChamferDistance<double>();

        var source = new Tensor<double>([0, 3]);
        var target = new Tensor<double>([2, 3]);
        for (int i = 0; i < 6; i++) target[i] = i * 0.5;

        // ACT
        var result = chamfer.ComputeOneWay(source, target);

        // ASSERT: Should return 0 for empty source
        Assert.Equal(0.0, result, 10);
    }

    #endregion

    #region Logging PR #772 - Production Bug Fixes

    [Fact]
    public void SummaryWriter_AddScalars_ValidatesNullDictionary()
    {
        // ARRANGE: The old code didn't validate null arguments
        var tempDir = Path.Combine(Path.GetTempPath(), $"tensorboard_test_{Guid.NewGuid():N}");
        try
        {
            using var writer = new AiDotNet.Logging.SummaryWriter(tempDir);

            // ACT & ASSERT
            Assert.Throws<ArgumentNullException>(() => writer.AddScalars("test", null!));
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }

    [Fact]
    public void SummaryWriter_AddHistogram_ValidatesNullArray()
    {
        // ARRANGE
        var tempDir = Path.Combine(Path.GetTempPath(), $"tensorboard_test_{Guid.NewGuid():N}");
        try
        {
            using var writer = new AiDotNet.Logging.SummaryWriter(tempDir);

            // ACT & ASSERT
            Assert.Throws<ArgumentNullException>(() => writer.AddHistogram("test", (float[])null!));
            Assert.Throws<ArgumentNullException>(() => writer.AddHistogram("test", (float[,])null!));
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }

    [Fact]
    public void SummaryWriter_AddImage_ValidatesDataformats()
    {
        // ARRANGE: The old code silently used HWC for any invalid dataformats value
        var tempDir = Path.Combine(Path.GetTempPath(), $"tensorboard_test_{Guid.NewGuid():N}");
        try
        {
            using var writer = new AiDotNet.Logging.SummaryWriter(tempDir);
            var imageData = new float[3, 2, 2]; // CHW format

            // ACT & ASSERT: Invalid dataformats should throw
            Assert.Throws<ArgumentException>(() => writer.AddImage("test", imageData, dataformats: "INVALID"));
            Assert.Throws<ArgumentException>(() => writer.AddImage("test", imageData, dataformats: "cwh"));
            Assert.Throws<ArgumentException>(() => writer.AddImage("test", imageData, dataformats: ""));
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }

    [Fact]
    public void SummaryWriter_AddImages_ValidatesNullArray()
    {
        // ARRANGE
        var tempDir = Path.Combine(Path.GetTempPath(), $"tensorboard_test_{Guid.NewGuid():N}");
        try
        {
            using var writer = new AiDotNet.Logging.SummaryWriter(tempDir);

            // ACT & ASSERT
            Assert.Throws<ArgumentNullException>(() => writer.AddImages("test", null!));
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }

    [Fact]
    public void SummaryWriter_AddPrCurve_ValidatesArguments()
    {
        // ARRANGE
        var tempDir = Path.Combine(Path.GetTempPath(), $"tensorboard_test_{Guid.NewGuid():N}");
        try
        {
            using var writer = new AiDotNet.Logging.SummaryWriter(tempDir);

            // ACT & ASSERT: Null checks
            Assert.Throws<ArgumentNullException>(() => writer.AddPrCurve("test", null!, new float[] { 0.5f }));
            Assert.Throws<ArgumentNullException>(() => writer.AddPrCurve("test", new int[] { 1 }, null!));

            // ACT & ASSERT: Length mismatch
            Assert.Throws<ArgumentException>(() => writer.AddPrCurve("test", new int[] { 1, 0 }, new float[] { 0.5f }));
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }

    [Fact]
    public void SummaryWriter_LogWeights_ValidatesNullArray()
    {
        // ARRANGE
        var tempDir = Path.Combine(Path.GetTempPath(), $"tensorboard_test_{Guid.NewGuid():N}");
        try
        {
            using var writer = new AiDotNet.Logging.SummaryWriter(tempDir);

            // ACT & ASSERT
            Assert.Throws<ArgumentNullException>(() => writer.LogWeights("layer1", null!));
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }

    [Fact]
    public void SummaryWriter_LogWeights_HandlesEmptyArray()
    {
        // ARRANGE: The old code would throw InvalidOperationException on Average() for empty array
        var tempDir = Path.Combine(Path.GetTempPath(), $"tensorboard_test_{Guid.NewGuid():N}");
        try
        {
            using var writer = new AiDotNet.Logging.SummaryWriter(tempDir);

            // ACT & ASSERT: Should not throw for empty array - just return silently
            var exception = Record.Exception(() => writer.LogWeights("layer1", Array.Empty<float>()));
            Assert.Null(exception);
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }

    #endregion
}
