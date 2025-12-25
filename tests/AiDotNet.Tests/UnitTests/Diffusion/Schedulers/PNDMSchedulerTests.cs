using AiDotNet.NeuralNetworks.Diffusion.Schedulers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion.Schedulers;

/// <summary>
/// Comprehensive unit tests for <see cref="PNDMScheduler{T}"/>.
/// </summary>
public class PNDMSchedulerTests
{
    #region Construction Tests

    [Fact]
    public void Constructor_WithValidConfig_CreatesScheduler()
    {
        // Arrange
        var config = SchedulerConfig<double>.CreateDefault();

        // Act
        var scheduler = new PNDMScheduler<double>(config);

        // Assert
        Assert.NotNull(scheduler);
        Assert.Equal(1000, scheduler.TrainTimesteps);
    }

    [Fact]
    public void Constructor_WithNullConfig_ThrowsArgumentNullException()
    {
        // Arrange & Act & Assert
        Assert.Throws<ArgumentNullException>(() => new PNDMScheduler<double>(null!));
    }

    #endregion

    #region SetTimesteps Tests

    [Fact]
    public void SetTimesteps_WithValidSteps_ProducesDescendingSequence()
    {
        // Arrange
        var scheduler = new PNDMScheduler<double>(
            new SchedulerConfig<double>(trainTimesteps: 100, betaStart: 0.0001, betaEnd: 0.02));

        // Act
        scheduler.SetTimesteps(20);
        var timesteps = scheduler.Timesteps;

        // Assert
        Assert.True(timesteps.Length <= 20 && timesteps.Length > 0);
        for (int i = 1; i < timesteps.Length; i++)
        {
            Assert.True(timesteps[i] < timesteps[i - 1],
                $"Timestep at index {i} ({timesteps[i]}) should be less than previous ({timesteps[i - 1]})");
        }
    }

    [Theory]
    [InlineData(10)]
    [InlineData(25)]
    [InlineData(50)]
    public void SetTimesteps_WithVariousSteps_ProducesCorrectLength(int inferenceSteps)
    {
        // Arrange
        var scheduler = new PNDMScheduler<double>(
            new SchedulerConfig<double>(trainTimesteps: 1000, betaStart: 0.0001, betaEnd: 0.02));

        // Act
        scheduler.SetTimesteps(inferenceSteps);

        // Assert
        Assert.True(scheduler.Timesteps.Length <= inferenceSteps);
        Assert.True(scheduler.Timesteps.Length > 0);
    }

    [Fact]
    public void SetTimesteps_ResetsInternalState()
    {
        // Arrange
        var scheduler = new PNDMScheduler<double>(SchedulerConfig<double>.CreateDefault());
        scheduler.SetTimesteps(20);

        var sample = new Vector<double>(new double[] { 0.1, 0.2, 0.3 });
        var modelOutput = new Vector<double>(new double[] { 0.01, 0.02, 0.03 });

        // Perform some steps
        foreach (var t in scheduler.Timesteps.Take(5))
        {
            scheduler.Step(modelOutput, t, sample, eta: 0.0);
        }

        // Act - Reset with new timesteps
        scheduler.SetTimesteps(10);

        // Assert - State should have been returned (get state to verify counter reset)
        var state = scheduler.GetState();
        Assert.Equal(0, state["counter"]);
    }

    #endregion

    #region Step Tests - PRK Phase

    [Fact]
    public void Step_InPrkPhase_ReturnsFiniteValues()
    {
        // Arrange
        var scheduler = new PNDMScheduler<double>(
            new SchedulerConfig<double>(trainTimesteps: 100, betaStart: 0.0001, betaEnd: 0.02));
        scheduler.SetTimesteps(20);

        var sample = new Vector<double>(new double[] { 0.1, -0.2, 0.3, -0.4 });
        var eps = new Vector<double>(new double[] { 0.05, 0.02, -0.01, -0.03 });

        // Act - First 4 steps are PRK mode
        for (int i = 0; i < 4; i++)
        {
            int t = scheduler.Timesteps[Math.Min(i, scheduler.Timesteps.Length - 1)];
            var result = scheduler.Step(eps, t, sample, eta: 0.0);

            // Assert
            Assert.Equal(sample.Length, result.Length);
            foreach (var v in result)
            {
                Assert.False(double.IsNaN(v), $"Result contains NaN at PRK step {i}");
                Assert.False(double.IsInfinity(v), $"Result contains Infinity at PRK step {i}");
            }
            sample = result;
        }
    }

    [Fact]
    public void Step_First4StepsArePrkMode()
    {
        // Arrange
        var scheduler = new PNDMScheduler<double>(
            new SchedulerConfig<double>(trainTimesteps: 100, betaStart: 0.0001, betaEnd: 0.02));
        scheduler.SetTimesteps(20);

        var sample = new Vector<double>(new double[] { 0.1, 0.2 });
        var eps = new Vector<double>(new double[] { 0.01, 0.02 });

        // Act & Assert - Track state counter to verify PRK mode
        for (int i = 0; i < 4; i++)
        {
            int t = scheduler.Timesteps[Math.Min(i, scheduler.Timesteps.Length - 1)];
            var state = scheduler.GetState();
            Assert.Equal(i, (int)state["counter"]);

            sample = scheduler.Step(eps, t, sample, eta: 0.0);
        }

        // After 4 steps, counter should be 4 (PLMS mode begins)
        var finalState = scheduler.GetState();
        Assert.Equal(4, (int)finalState["counter"]);
    }

    #endregion

    #region Step Tests - PLMS Phase

    [Fact]
    public void Step_InPlmsPhase_ReturnsFiniteValues()
    {
        // Arrange
        var scheduler = new PNDMScheduler<double>(
            new SchedulerConfig<double>(trainTimesteps: 100, betaStart: 0.0001, betaEnd: 0.02));
        scheduler.SetTimesteps(20);

        var sample = new Vector<double>(new double[] { 0.1, -0.2, 0.3, -0.4 });
        var eps = new Vector<double>(new double[] { 0.05, 0.02, -0.01, -0.03 });

        // First pass through PRK phase
        for (int i = 0; i < 4 && i < scheduler.Timesteps.Length; i++)
        {
            sample = scheduler.Step(eps, scheduler.Timesteps[i], sample, eta: 0.0);
        }

        // Act - Continue in PLMS phase
        for (int i = 4; i < Math.Min(10, scheduler.Timesteps.Length); i++)
        {
            var result = scheduler.Step(eps, scheduler.Timesteps[i], sample, eta: 0.0);

            // Assert
            Assert.Equal(sample.Length, result.Length);
            foreach (var v in result)
            {
                Assert.False(double.IsNaN(v), $"Result contains NaN at PLMS step {i}");
                Assert.False(double.IsInfinity(v), $"Result contains Infinity at PLMS step {i}");
            }
            sample = result;
        }
    }

    [Fact]
    public void Step_PlmsPhaseUsesHistoryOfPredictions()
    {
        // Arrange
        var scheduler = new PNDMScheduler<double>(
            new SchedulerConfig<double>(trainTimesteps: 100, betaStart: 0.0001, betaEnd: 0.02));
        scheduler.SetTimesteps(20);

        var sample = new Vector<double>(new double[] { 0.5, 0.5 });

        // Go through PRK phase with consistent predictions
        for (int i = 0; i < 4 && i < scheduler.Timesteps.Length; i++)
        {
            var eps = new Vector<double>(new double[] { 0.1, 0.1 });
            sample = scheduler.Step(eps, scheduler.Timesteps[i], sample, eta: 0.0);
        }

        // In PLMS phase, state should track history
        var state = scheduler.GetState();
        Assert.True((int)state["ets_count"] > 0, "PLMS should maintain history of predictions");
    }

    #endregion

    #region Step Tests - Validation

    [Fact]
    public void Step_WithNullModelOutput_ThrowsArgumentNullException()
    {
        // Arrange
        var scheduler = new PNDMScheduler<double>(SchedulerConfig<double>.CreateDefault());
        scheduler.SetTimesteps(10);
        var sample = new Vector<double>(new double[] { 0.1, 0.2 });

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            scheduler.Step(null!, scheduler.Timesteps[0], sample, 0.0));
    }

    [Fact]
    public void Step_WithNullSample_ThrowsArgumentNullException()
    {
        // Arrange
        var scheduler = new PNDMScheduler<double>(SchedulerConfig<double>.CreateDefault());
        scheduler.SetTimesteps(10);
        var modelOutput = new Vector<double>(new double[] { 0.1, 0.2 });

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            scheduler.Step(modelOutput, scheduler.Timesteps[0], null!, 0.0));
    }

    [Fact]
    public void Step_WithMismatchedLengths_ThrowsArgumentException()
    {
        // Arrange
        var scheduler = new PNDMScheduler<double>(SchedulerConfig<double>.CreateDefault());
        scheduler.SetTimesteps(10);
        var modelOutput = new Vector<double>(new double[] { 0.1, 0.2 });
        var sample = new Vector<double>(new double[] { 0.1, 0.2, 0.3 });

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            scheduler.Step(modelOutput, scheduler.Timesteps[0], sample, 0.0));
    }

    #endregion

    #region Step Tests - Deterministic Behavior

    [Fact]
    public void Step_IsDeterministic()
    {
        // Arrange
        var config = new SchedulerConfig<double>(trainTimesteps: 100, betaStart: 0.0001, betaEnd: 0.02);

        // Run 1
        var scheduler1 = new PNDMScheduler<double>(config);
        scheduler1.SetTimesteps(10);
        var sample1 = new Vector<double>(new double[] { 0.1, 0.2, 0.3 });
        var eps1 = new Vector<double>(new double[] { 0.05, 0.02, -0.01 });

        var results1 = new List<Vector<double>>();
        foreach (var t in scheduler1.Timesteps)
        {
            sample1 = scheduler1.Step(eps1, t, sample1, eta: 0.0);
            results1.Add(sample1);
        }

        // Run 2
        var scheduler2 = new PNDMScheduler<double>(config);
        scheduler2.SetTimesteps(10);
        var sample2 = new Vector<double>(new double[] { 0.1, 0.2, 0.3 });
        var eps2 = new Vector<double>(new double[] { 0.05, 0.02, -0.01 });

        var results2 = new List<Vector<double>>();
        foreach (var t in scheduler2.Timesteps)
        {
            sample2 = scheduler2.Step(eps2, t, sample2, eta: 0.0);
            results2.Add(sample2);
        }

        // Assert - Both runs should produce identical results
        Assert.Equal(results1.Count, results2.Count);
        for (int i = 0; i < results1.Count; i++)
        {
            for (int j = 0; j < results1[i].Length; j++)
            {
                Assert.Equal(results1[i][j], results2[i][j], precision: 12);
            }
        }
    }

    #endregion

    #region State Management Tests

    [Fact]
    public void GetState_ReturnsCounterAndEtsCount()
    {
        // Arrange
        var scheduler = new PNDMScheduler<double>(SchedulerConfig<double>.CreateDefault());
        scheduler.SetTimesteps(20);

        // Act
        var state = scheduler.GetState();

        // Assert
        Assert.True(state.TryGetValue("counter", out var counter));
        Assert.True(state.TryGetValue("ets_count", out var etsCount));
        Assert.Equal(0, (int)counter!);
        Assert.Equal(0, (int)etsCount!);
    }

    [Fact]
    public void GetState_AfterSteps_ReflectsProgress()
    {
        // Arrange
        var scheduler = new PNDMScheduler<double>(SchedulerConfig<double>.CreateDefault());
        scheduler.SetTimesteps(20);

        var sample = new Vector<double>(new double[] { 0.1, 0.2 });
        var eps = new Vector<double>(new double[] { 0.01, 0.02 });

        // Perform 5 steps
        for (int i = 0; i < 5 && i < scheduler.Timesteps.Length; i++)
        {
            sample = scheduler.Step(eps, scheduler.Timesteps[i], sample, eta: 0.0);
        }

        // Act
        var state = scheduler.GetState();

        // Assert
        Assert.Equal(5, (int)state["counter"]);
        Assert.True((int)state["ets_count"] > 0);
    }

    [Fact]
    public void LoadState_ResetsCounterToZero()
    {
        // Arrange
        var scheduler1 = new PNDMScheduler<double>(SchedulerConfig<double>.CreateDefault());
        scheduler1.SetTimesteps(20);

        var sample = new Vector<double>(new double[] { 0.1, 0.2 });
        var eps = new Vector<double>(new double[] { 0.01, 0.02 });

        // Perform some steps
        for (int i = 0; i < 3 && i < scheduler1.Timesteps.Length; i++)
        {
            sample = scheduler1.Step(eps, scheduler1.Timesteps[i], sample, eta: 0.0);
        }

        var state = scheduler1.GetState();

        // Act
        var scheduler2 = new PNDMScheduler<double>(SchedulerConfig<double>.CreateDefault());
        scheduler2.SetTimesteps(20);
        scheduler2.LoadState(state);

        // Assert - Counter is NOT restored because _ets history cannot be serialized.
        // Restoring counter without history would cause PLMS to fail due to missing predictions.
        var restoredState = scheduler2.GetState();
        Assert.Equal(0, (int)restoredState["counter"]);
        Assert.Equal(0, (int)restoredState["ets_count"]);
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void FullDenoisingLoop_ProducesFiniteResults()
    {
        // Arrange
        var scheduler = new PNDMScheduler<double>(SchedulerConfig<double>.CreateDefault());
        scheduler.SetTimesteps(25);

        var sample = new Vector<double>(new double[64]);
        var random = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < sample.Length; i++)
        {
            sample[i] = random.NextDouble() * 2 - 1;
        }

        // Act - Simulate a full denoising loop
        foreach (var t in scheduler.Timesteps)
        {
            // Simulate model output (random noise prediction)
            var modelOutput = new Vector<double>(sample.Length);
            for (int i = 0; i < modelOutput.Length; i++)
            {
                modelOutput[i] = random.NextDouble() * 0.1 - 0.05;
            }

            sample = scheduler.Step(modelOutput, t, sample, eta: 0.0);
        }

        // Assert - Final sample should be finite
        foreach (var v in sample)
        {
            Assert.False(double.IsNaN(v), "Final sample contains NaN");
            Assert.False(double.IsInfinity(v), "Final sample contains Infinity");
        }
    }

    [Fact]
    public void FullDenoisingLoop_WithFewSteps_WorksCorrectly()
    {
        // Arrange - PNDM is designed for fast inference with few steps
        var scheduler = new PNDMScheduler<double>(SchedulerConfig<double>.CreateDefault());
        scheduler.SetTimesteps(10); // Very few steps

        var sample = new Vector<double>(new double[] { 0.5, -0.5, 0.25, -0.25 });

        // Act
        foreach (var t in scheduler.Timesteps)
        {
            var eps = new Vector<double>(new double[] { 0.01, -0.01, 0.005, -0.005 });
            sample = scheduler.Step(eps, t, sample, eta: 0.0);
        }

        // Assert
        foreach (var v in sample)
        {
            Assert.False(double.IsNaN(v));
            Assert.False(double.IsInfinity(v));
        }
    }

    #endregion

    #region Comparison with DDIM

    [Fact]
    public void PNDMAndDDIM_ProduceDifferentResults()
    {
        // Arrange
        var config = new SchedulerConfig<double>(trainTimesteps: 100, betaStart: 0.0001, betaEnd: 0.02);
        var pndm = new PNDMScheduler<double>(config);
        var ddim = new DDIMScheduler<double>(config);

        pndm.SetTimesteps(20);
        ddim.SetTimesteps(20);

        var sample = new Vector<double>(new double[] { 0.1, 0.2, 0.3 });
        var eps = new Vector<double>(new double[] { 0.01, 0.02, 0.03 });

        // Act
        var pndmSample = new Vector<double>(sample.Length);
        var ddimSample = new Vector<double>(sample.Length);
        for (int i = 0; i < sample.Length; i++)
        {
            pndmSample[i] = sample[i];
            ddimSample[i] = sample[i];
        }

        foreach (var t in pndm.Timesteps)
        {
            pndmSample = pndm.Step(eps, t, pndmSample, eta: 0.0);
        }

        foreach (var t in ddim.Timesteps)
        {
            ddimSample = ddim.Step(eps, t, ddimSample, eta: 0.0);
        }

        // Assert - Results should differ (different algorithms)
        bool anyDiff = false;
        for (int i = 0; i < sample.Length; i++)
        {
            if (Math.Abs(pndmSample[i] - ddimSample[i]) > 1e-9)
            {
                anyDiff = true;
                break;
            }
        }
        Assert.True(anyDiff, "PNDM and DDIM should produce different results");
    }

    #endregion
}
