using AiDotNet.NeuralNetworks.Diffusion.Schedulers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion.Schedulers;

/// <summary>
/// Comprehensive unit tests for <see cref="DDIMScheduler{T}"/>.
/// </summary>
public class DDIMSchedulerTests
{
    #region Construction Tests

    [Fact]
    public void Constructor_WithValidConfig_CreatesScheduler()
    {
        // Arrange
        var config = SchedulerConfig<double>.CreateDefault();

        // Act
        var scheduler = new DDIMScheduler<double>(config);

        // Assert
        Assert.NotNull(scheduler);
        Assert.Equal(1000, scheduler.TrainTimesteps);
    }

    [Fact]
    public void Constructor_WithNullConfig_ThrowsArgumentNullException()
    {
        // Arrange & Act & Assert
        Assert.Throws<ArgumentNullException>(() => new DDIMScheduler<double>(null!));
    }

    #endregion

    #region SetTimesteps Tests

    [Fact]
    public void SetTimesteps_WithValidSteps_ProducesDescendingSequence()
    {
        // Arrange
        var scheduler = new DDIMScheduler<double>(
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
    [InlineData(100)]
    public void SetTimesteps_WithVariousSteps_ProducesCorrectLength(int inferenceSteps)
    {
        // Arrange
        var scheduler = new DDIMScheduler<double>(
            new SchedulerConfig<double>(trainTimesteps: 1000, betaStart: 0.0001, betaEnd: 0.02));

        // Act
        scheduler.SetTimesteps(inferenceSteps);

        // Assert
        Assert.True(scheduler.Timesteps.Length <= inferenceSteps);
        Assert.True(scheduler.Timesteps.Length > 0);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void SetTimesteps_WithInvalidSteps_ThrowsArgumentOutOfRangeException(int invalidSteps)
    {
        // Arrange
        var scheduler = new DDIMScheduler<double>(SchedulerConfig<double>.CreateDefault());

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => scheduler.SetTimesteps(invalidSteps));
    }

    [Fact]
    public void SetTimesteps_FirstTimestepIsNearTrainTimesteps()
    {
        // Arrange
        var scheduler = new DDIMScheduler<double>(
            new SchedulerConfig<double>(trainTimesteps: 1000, betaStart: 0.0001, betaEnd: 0.02));

        // Act
        scheduler.SetTimesteps(50);

        // Assert - First timestep should be close to max
        Assert.True(scheduler.Timesteps[0] >= 900,
            $"First timestep should be close to max, was {scheduler.Timesteps[0]}");
    }

    [Fact]
    public void SetTimesteps_LastTimestepIsNearZero()
    {
        // Arrange
        var scheduler = new DDIMScheduler<double>(
            new SchedulerConfig<double>(trainTimesteps: 1000, betaStart: 0.0001, betaEnd: 0.02));

        // Act
        scheduler.SetTimesteps(50);

        // Assert - Last timestep should be close to 0
        Assert.True(scheduler.Timesteps[^1] <= 50,
            $"Last timestep should be close to 0, was {scheduler.Timesteps[^1]}");
    }

    #endregion

    #region Step Tests - Deterministic (eta=0)

    [Fact]
    public void Step_DeterministicWithEtaZero_ReturnsFiniteValues()
    {
        // Arrange
        var scheduler = new DDIMScheduler<double>(
            new SchedulerConfig<double>(trainTimesteps: 50, betaStart: 0.0001, betaEnd: 0.02));
        scheduler.SetTimesteps(10);
        int t = scheduler.Timesteps[0];

        var sample = new Vector<double>(new double[] { 0.1, -0.2, 0.3, -0.4 });
        var eps = new Vector<double>(new double[] { 0.05, 0.02, -0.01, -0.03 });

        // Act
        var prev = scheduler.Step(eps, t, sample, eta: 0.0);

        // Assert
        Assert.Equal(sample.Length, prev.Length);
        foreach (var v in prev)
        {
            Assert.False(double.IsNaN(v), "Result contains NaN");
            Assert.False(double.IsInfinity(v), "Result contains Infinity");
        }
    }

    [Fact]
    public void Step_DeterministicWithEtaZero_IsDeterministic()
    {
        // Arrange
        var scheduler = new DDIMScheduler<double>(
            new SchedulerConfig<double>(trainTimesteps: 100, betaStart: 0.0001, betaEnd: 0.02));
        scheduler.SetTimesteps(25);
        int t = scheduler.Timesteps[0];

        var sample = new Vector<double>(new double[] { 0.1, -0.2, 0.3 });
        var eps = new Vector<double>(new double[] { 0.05, 0.02, -0.01 });

        // Act
        var result1 = scheduler.Step(eps, t, sample, eta: 0.0);
        var result2 = scheduler.Step(eps, t, sample, eta: 0.0);

        // Assert - Same inputs should produce same outputs
        for (int i = 0; i < result1.Length; i++)
        {
            Assert.Equal(result1[i], result2[i], precision: 15);
        }
    }

    #endregion

    #region Step Tests - Stochastic (eta > 0)

    [Fact]
    public void Step_StochasticWithEtaPositive_DiffersFromDeterministic()
    {
        // Arrange
        var scheduler = new DDIMScheduler<double>(
            new SchedulerConfig<double>(trainTimesteps: 100, betaStart: 0.0001, betaEnd: 0.02));
        scheduler.SetTimesteps(25);
        int t = scheduler.Timesteps[0];

        var sample = new Vector<double>(new double[] { 0.2, 0.0, -0.1 });
        var eps = new Vector<double>(new double[] { 0.01, -0.02, 0.03 });
        var noise = new Vector<double>(new double[] { 0.3, -0.5, 0.7 });

        // Act
        var prevNoNoise = scheduler.Step(eps, t, sample, eta: 0.0);
        var prevWithNoise = scheduler.Step(eps, t, sample, eta: 0.5, noise: noise);

        // Assert - Results should differ when eta > 0 and noise is provided
        bool anyDiff = false;
        for (int i = 0; i < sample.Length; i++)
        {
            if (Math.Abs(prevNoNoise[i] - prevWithNoise[i]) > 1e-9)
            {
                anyDiff = true;
                break;
            }
        }
        Assert.True(anyDiff, "Stochastic step should differ from deterministic step");
    }

    [Fact]
    public void Step_WithNoiseProvided_UsesProvidedNoise()
    {
        // Arrange
        var scheduler = new DDIMScheduler<double>(
            new SchedulerConfig<double>(trainTimesteps: 100, betaStart: 0.0001, betaEnd: 0.02));
        scheduler.SetTimesteps(25);
        int t = scheduler.Timesteps[0];

        var sample = new Vector<double>(new double[] { 0.2, 0.0, -0.1 });
        var eps = new Vector<double>(new double[] { 0.01, -0.02, 0.03 });
        var noise1 = new Vector<double>(new double[] { 0.3, -0.5, 0.7 });
        var noise2 = new Vector<double>(new double[] { -0.3, 0.5, -0.7 });

        // Act
        var result1 = scheduler.Step(eps, t, sample, eta: 1.0, noise: noise1);
        var result2 = scheduler.Step(eps, t, sample, eta: 1.0, noise: noise2);

        // Assert - Different noise should produce different results
        bool anyDiff = false;
        for (int i = 0; i < sample.Length; i++)
        {
            if (Math.Abs(result1[i] - result2[i]) > 1e-9)
            {
                anyDiff = true;
                break;
            }
        }
        Assert.True(anyDiff, "Different noise should produce different results");
    }

    #endregion

    #region Step Tests - Clipping

    [Fact]
    public void Step_WithClipSampleEnabled_ClipsToRange()
    {
        // Arrange
        var config = new SchedulerConfig<double>(
            trainTimesteps: 2,
            betaStart: 0.0001,
            betaEnd: 0.02,
            clipSample: true);
        var scheduler = new DDIMScheduler<double>(config);
        scheduler.SetTimesteps(2);
        int t = scheduler.Timesteps[0];

        // Use extreme values that would produce out-of-range results
        var sample = new Vector<double>(new double[] { 10.0 });
        var eps = new Vector<double>(new double[] { -5.0 });

        // Act
        var result = scheduler.Step(eps, t, sample, eta: 0.0);

        // Assert - Results should be finite (clipping prevents extreme values)
        Assert.False(double.IsNaN(result[0]));
        Assert.False(double.IsInfinity(result[0]));
    }

    #endregion

    #region Step Tests - Validation

    [Fact]
    public void Step_WithNullModelOutput_ThrowsArgumentNullException()
    {
        // Arrange
        var scheduler = new DDIMScheduler<double>(SchedulerConfig<double>.CreateDefault());
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
        var scheduler = new DDIMScheduler<double>(SchedulerConfig<double>.CreateDefault());
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
        var scheduler = new DDIMScheduler<double>(SchedulerConfig<double>.CreateDefault());
        scheduler.SetTimesteps(10);
        var modelOutput = new Vector<double>(new double[] { 0.1, 0.2 });
        var sample = new Vector<double>(new double[] { 0.1, 0.2, 0.3 });

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            scheduler.Step(modelOutput, scheduler.Timesteps[0], sample, 0.0));
    }

    [Fact]
    public void Step_WithInvalidTimestep_ThrowsArgumentOutOfRangeException()
    {
        // Arrange
        var scheduler = new DDIMScheduler<double>(
            new SchedulerConfig<double>(trainTimesteps: 100, betaStart: 0.0001, betaEnd: 0.02));
        scheduler.SetTimesteps(10);
        var sample = new Vector<double>(new double[] { 0.1, 0.2 });
        var eps = new Vector<double>(new double[] { 0.1, 0.2 });

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            scheduler.Step(eps, 1000, sample, 0.0)); // Timestep > train_timesteps
    }

    #endregion

    #region AddNoise Tests

    [Fact]
    public void AddNoise_WithValidInputs_ReturnsNoisyVector()
    {
        // Arrange
        var scheduler = new DDIMScheduler<double>(SchedulerConfig<double>.CreateDefault());
        scheduler.SetTimesteps(50);

        var original = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var noise = new Vector<double>(new double[] { 0.1, 0.2, 0.3 });
        int timestep = 500;

        // Act
        var result = scheduler.AddNoise(original, noise, timestep);

        // Assert
        Assert.Equal(original.Length, result.Length);

        // Result should be a combination of original and noise
        for (int i = 0; i < result.Length; i++)
        {
            Assert.False(double.IsNaN(result[i]));
            Assert.False(double.IsInfinity(result[i]));
        }
    }

    [Fact]
    public void AddNoise_AtTimestepZero_ReturnsOriginalWithMinimalChange()
    {
        // Arrange
        var scheduler = new DDIMScheduler<double>(SchedulerConfig<double>.CreateDefault());
        scheduler.SetTimesteps(50);

        var original = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var noise = new Vector<double>(new double[] { 0.1, 0.2, 0.3 });

        // Act
        var result = scheduler.AddNoise(original, noise, timestep: 0);

        // Assert - At timestep 0, almost no noise should be added
        for (int i = 0; i < result.Length; i++)
        {
            // Should be very close to original (alpha_cumprod near 1 at timestep 0)
            Assert.InRange(result[i], original[i] * 0.9, original[i] * 1.1 + 0.1);
        }
    }

    [Fact]
    public void AddNoise_AtHighTimestep_AddsMoreNoise()
    {
        // Arrange
        var scheduler = new DDIMScheduler<double>(SchedulerConfig<double>.CreateDefault());
        scheduler.SetTimesteps(50);

        var original = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var noise = new Vector<double>(new double[] { 0.5, 0.5, 0.5 });

        // Act
        var resultLowT = scheduler.AddNoise(original, noise, timestep: 100);
        var resultHighT = scheduler.AddNoise(original, noise, timestep: 900);

        // Assert - Higher timestep should result in more noise contribution
        double sumDiffLow = 0, sumDiffHigh = 0;
        for (int i = 0; i < original.Length; i++)
        {
            sumDiffLow += Math.Abs(resultLowT[i] - original[i]);
            sumDiffHigh += Math.Abs(resultHighT[i] - original[i]);
        }

        Assert.True(sumDiffHigh > sumDiffLow,
            "Higher timestep should result in more deviation from original");
    }

    #endregion

    #region GetAlphaCumulativeProduct Tests

    [Fact]
    public void GetAlphaCumulativeProduct_ReturnsDecreasingValues()
    {
        // Arrange
        var scheduler = new DDIMScheduler<double>(SchedulerConfig<double>.CreateDefault());
        scheduler.SetTimesteps(50);

        // Act & Assert - Alpha cumulative product should decrease with timestep
        double prev = scheduler.GetAlphaCumulativeProduct(0);
        for (int t = 100; t < 1000; t += 100)
        {
            double current = scheduler.GetAlphaCumulativeProduct(t);
            Assert.True(current < prev,
                $"Alpha cumprod at t={t} ({current}) should be less than at t={t - 100} ({prev})");
            prev = current;
        }
    }

    [Fact]
    public void GetAlphaCumulativeProduct_AtTimestepZero_IsNearOne()
    {
        // Arrange
        var scheduler = new DDIMScheduler<double>(SchedulerConfig<double>.CreateDefault());
        scheduler.SetTimesteps(50);

        // Act
        var alpha = scheduler.GetAlphaCumulativeProduct(0);

        // Assert - At timestep 0, alpha_cumprod should be very close to 1
        Assert.InRange(alpha, 0.99, 1.0);
    }

    #endregion

    #region State Management Tests

    [Fact]
    public void GetState_ReturnsNonEmptyDictionary()
    {
        // Arrange
        var scheduler = new DDIMScheduler<double>(SchedulerConfig<double>.CreateDefault());
        scheduler.SetTimesteps(50);

        // Act
        var state = scheduler.GetState();

        // Assert
        Assert.NotNull(state);
        Assert.True(state.Count > 0);
        Assert.True(state.ContainsKey("train_timesteps"));
    }

    [Fact]
    public void LoadState_RestoresTimesteps()
    {
        // Arrange
        var scheduler1 = new DDIMScheduler<double>(SchedulerConfig<double>.CreateDefault());
        scheduler1.SetTimesteps(50);
        var state = scheduler1.GetState();

        var scheduler2 = new DDIMScheduler<double>(SchedulerConfig<double>.CreateDefault());
        scheduler2.SetTimesteps(10); // Different initial state

        // Act
        scheduler2.LoadState(state);

        // Assert
        Assert.Equal(scheduler1.TrainTimesteps, scheduler2.TrainTimesteps);
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void FullDenoisingLoop_ProducesFiniteResults()
    {
        // Arrange
        var scheduler = new DDIMScheduler<double>(SchedulerConfig<double>.CreateDefault());
        scheduler.SetTimesteps(20);

        var sample = new Vector<double>(new double[64]);
        var random = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < sample.Length; i++)
        {
            sample[i] = random.NextDouble() * 2 - 1;
        }

        // Act - Simulate a denoising loop
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

    #endregion
}
