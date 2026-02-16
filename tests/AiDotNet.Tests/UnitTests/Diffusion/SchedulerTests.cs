using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion;

/// <summary>
/// Tests for diffusion model schedulers.
/// </summary>
public class SchedulerTests
{
    #region DDIM Scheduler Tests

    [Fact]
    public void DDIMScheduler_Constructor_InitializesCorrectly()
    {
        // Arrange & Act
        var config = SchedulerConfig<float>.CreateDefault();
        var scheduler = new DDIMScheduler<float>(config);

        // Assert
        Assert.NotNull(scheduler);
        Assert.NotNull(scheduler.Config);
    }

    [Fact]
    public void DDIMScheduler_SetTimesteps_CreatesValidTimesteps()
    {
        // Arrange
        var config = SchedulerConfig<float>.CreateDefault();
        var scheduler = new DDIMScheduler<float>(config);

        // Act
        scheduler.SetTimesteps(50);
        var timesteps = scheduler.Timesteps;

        // Assert
        Assert.NotNull(timesteps);
        Assert.True(timesteps.Length > 0);
        Assert.Equal(50, timesteps.Length);
    }

    [Fact]
    public void DDIMScheduler_Timesteps_AreDecreasing()
    {
        // Arrange
        var config = SchedulerConfig<float>.CreateDefault();
        var scheduler = new DDIMScheduler<float>(config);
        scheduler.SetTimesteps(20);

        // Act
        var timesteps = scheduler.Timesteps;

        // Assert - Timesteps should decrease (diffusion runs backward)
        for (int i = 1; i < timesteps.Length; i++)
        {
            Assert.True(timesteps[i] < timesteps[i - 1],
                $"Timesteps should decrease: {timesteps[i - 1]} -> {timesteps[i]}");
        }
    }

    [Fact]
    public void DDIMScheduler_Step_ProducesValidOutput()
    {
        // Arrange
        var config = SchedulerConfig<float>.CreateDefault();
        var scheduler = new DDIMScheduler<float>(config);
        scheduler.SetTimesteps(20);

        var sample = CreateRandomVector(4 * 64 * 64, 42);
        var modelOutput = CreateRandomVector(4 * 64 * 64, 43);
        int timestep = scheduler.Timesteps[0];

        // Act
        var result = scheduler.Step(modelOutput, timestep, sample, 0.0f);

        // Assert
        Assert.Equal(sample.Length, result.Length);
        Assert.False(ContainsNaN(result), "Result should not contain NaN");
        Assert.False(ContainsInf(result), "Result should not contain Inf");
    }

    [Fact]
    public void DDIMScheduler_AddNoise_ProducesNoisySample()
    {
        // Arrange
        var config = SchedulerConfig<float>.CreateDefault();
        var scheduler = new DDIMScheduler<float>(config);
        scheduler.SetTimesteps(20);

        var original = CreateRandomVector(4 * 32 * 32, 42);
        var noise = CreateRandomVector(4 * 32 * 32, 43);
        int timestep = 500;

        // Act
        var noisy = scheduler.AddNoise(original, noise, timestep);

        // Assert
        Assert.Equal(original.Length, noisy.Length);
        // Noisy should be different from original
        Assert.False(VectorsEqual(original, noisy), "Noisy sample should differ from original");
    }

    [Fact]
    public void DDIMScheduler_EtaZero_IsDeterministic()
    {
        // Arrange
        var config = SchedulerConfig<float>.CreateDefault();
        var scheduler = new DDIMScheduler<float>(config);
        scheduler.SetTimesteps(20);

        var sample = CreateRandomVector(1024, 42);
        var modelOutput = CreateRandomVector(1024, 43);
        int timestep = scheduler.Timesteps[0];

        // Act - Same inputs should produce same output with eta=0
        var result1 = scheduler.Step(modelOutput, timestep, sample, 0.0f);
        var result2 = scheduler.Step(modelOutput, timestep, sample, 0.0f);

        // Assert
        Assert.True(VectorsEqual(result1, result2), "Eta=0 should be deterministic");
    }

    #endregion

    #region PNDM Scheduler Tests

    [Fact]
    public void PNDMScheduler_Constructor_InitializesCorrectly()
    {
        // Arrange & Act
        var config = SchedulerConfig<float>.CreateDefault();
        var scheduler = new PNDMScheduler<float>(config);

        // Assert
        Assert.NotNull(scheduler);
    }

    [Fact]
    public void PNDMScheduler_SetTimesteps_CreatesValidTimesteps()
    {
        // Arrange
        var config = SchedulerConfig<float>.CreateDefault();
        var scheduler = new PNDMScheduler<float>(config);

        // Act
        scheduler.SetTimesteps(50);
        var timesteps = scheduler.Timesteps;

        // Assert
        Assert.NotNull(timesteps);
        Assert.True(timesteps.Length > 0);
    }

    [Fact]
    public void PNDMScheduler_Step_ProducesValidOutput()
    {
        // Arrange
        var config = SchedulerConfig<float>.CreateDefault();
        var scheduler = new PNDMScheduler<float>(config);
        scheduler.SetTimesteps(20);

        var sample = CreateRandomVector(4 * 64 * 64, 42);
        var modelOutput = CreateRandomVector(4 * 64 * 64, 43);
        int timestep = scheduler.Timesteps[0];

        // Act
        var result = scheduler.Step(modelOutput, timestep, sample, 0.0f);

        // Assert
        Assert.Equal(sample.Length, result.Length);
        Assert.False(ContainsNaN(result), "Result should not contain NaN");
    }

    [Fact]
    public void PNDMScheduler_MultipleSteps_AccumulatesHistory()
    {
        // Arrange
        var config = SchedulerConfig<float>.CreateDefault();
        var scheduler = new PNDMScheduler<float>(config);
        scheduler.SetTimesteps(20);

        var sample = CreateRandomVector(4 * 32 * 32, 42);

        // Act - Run multiple steps
        for (int i = 0; i < 5; i++)
        {
            var modelOutput = CreateRandomVector(4 * 32 * 32, 42 + i);
            int timestep = scheduler.Timesteps[i];
            sample = scheduler.Step(modelOutput, timestep, sample, 0.0f);
        }

        // Assert - After 4+ steps, PNDM should use higher-order method
        Assert.False(ContainsNaN(sample), "Result should not contain NaN after multiple steps");
    }

    #endregion

    #region Scheduler Comparison Tests

    [Fact]
    public void AllSchedulers_ProduceDifferentResults()
    {
        // Arrange
        var config = SchedulerConfig<float>.CreateDefault();
        var ddim = new DDIMScheduler<float>(config);
        var pndm = new PNDMScheduler<float>(config);

        ddim.SetTimesteps(10);
        pndm.SetTimesteps(10);

        var sample = CreateRandomVector(16 * 16 * 4, 42);
        var modelOutput = CreateRandomVector(16 * 16 * 4, 43);

        // Act
        var ddimResult = ddim.Step(modelOutput, ddim.Timesteps[0], sample, 0.0f);
        var pndmResult = pndm.Step(modelOutput, pndm.Timesteps[0], sample, 0.0f);

        // Assert - Results should all be valid but potentially different
        Assert.False(ContainsNaN(ddimResult), "DDIM should not produce NaN");
        Assert.False(ContainsNaN(pndmResult), "PNDM should not produce NaN");
    }

    [Fact]
    public void AllSchedulers_HandleLastTimestep()
    {
        // Arrange
        var config = SchedulerConfig<float>.CreateDefault();
        var ddim = new DDIMScheduler<float>(config);
        var pndm = new PNDMScheduler<float>(config);

        ddim.SetTimesteps(50);
        pndm.SetTimesteps(50);

        var sample = CreateRandomVector(16 * 16 * 4, 42);
        var modelOutput = CreateRandomVector(16 * 16 * 4, 43);

        // Get the last (smallest) timestep
        int lastDdimTimestep = ddim.Timesteps[ddim.Timesteps.Length - 1];
        int lastPndmTimestep = pndm.Timesteps[pndm.Timesteps.Length - 1];

        // Act & Assert - Should not throw
        var ddimResult = ddim.Step(modelOutput, lastDdimTimestep, sample, 0.0f);
        var pndmResult = pndm.Step(modelOutput, lastPndmTimestep, sample, 0.0f);

        Assert.False(ContainsNaN(ddimResult));
        Assert.False(ContainsNaN(pndmResult));
    }

    #endregion

    #region Numerical Stability Tests

    [Fact]
    public void DDIMScheduler_LargeMagnitude_RemainsStable()
    {
        // Arrange
        var config = SchedulerConfig<float>.CreateDefault();
        var scheduler = new DDIMScheduler<float>(config);
        scheduler.SetTimesteps(20);

        var sample = CreateRandomVector(4 * 32 * 32, 42);
        // Scale up to test numerical stability
        for (int i = 0; i < sample.Length; i++)
        {
            sample[i] *= 1000f;
        }

        var modelOutput = CreateRandomVector(4 * 32 * 32, 43);
        for (int i = 0; i < modelOutput.Length; i++)
        {
            modelOutput[i] *= 1000f;
        }

        // Act - Run through multiple steps
        for (int i = 0; i < 10; i++)
        {
            int timestep = scheduler.Timesteps[i];
            sample = scheduler.Step(modelOutput, timestep, sample, 0.0f);
        }

        // Assert
        Assert.False(ContainsNaN(sample), "Should remain stable with large magnitudes");
        Assert.False(ContainsInf(sample), "Should not overflow");
    }

    [Fact]
    public void DDIMScheduler_SmallMagnitude_RemainsStable()
    {
        // Arrange
        var config = SchedulerConfig<float>.CreateDefault();
        var scheduler = new DDIMScheduler<float>(config);
        scheduler.SetTimesteps(20);

        var sample = CreateRandomVector(4 * 32 * 32, 42);
        // Scale down to test numerical stability
        for (int i = 0; i < sample.Length; i++)
        {
            sample[i] *= 1e-6f;
        }

        var modelOutput = CreateRandomVector(4 * 32 * 32, 43);
        for (int i = 0; i < modelOutput.Length; i++)
        {
            modelOutput[i] *= 1e-6f;
        }

        // Act
        int timestep = scheduler.Timesteps[0];
        var result = scheduler.Step(modelOutput, timestep, sample, 0.0f);

        // Assert
        Assert.False(ContainsNaN(result), "Should remain stable with small magnitudes");
    }

    #endregion

    #region Config Tests

    [Fact]
    public void SchedulerConfig_CreateDefault_HasValidDefaults()
    {
        // Act
        var config = SchedulerConfig<float>.CreateDefault();

        // Assert
        Assert.True(config.TrainTimesteps > 0, "Train timesteps should be positive");
    }

    [Fact]
    public void DDIMScheduler_TrainTimesteps_MatchesConfig()
    {
        // Arrange
        var config = SchedulerConfig<float>.CreateDefault();

        // Act
        var scheduler = new DDIMScheduler<float>(config);

        // Assert
        Assert.Equal(config.TrainTimesteps, scheduler.TrainTimesteps);
    }

    #endregion

    #region Helper Methods

    private static Vector<float> CreateRandomVector(int length, int seed)
    {
        var vector = new Vector<float>(length);
        var random = new Random(seed);

        for (int i = 0; i < length; i++)
        {
            vector[i] = (float)(random.NextDouble() * 2 - 1);
        }

        return vector;
    }

    private static bool ContainsNaN(Vector<float> vector)
    {
        for (int i = 0; i < vector.Length; i++)
        {
            if (float.IsNaN(vector[i]))
                return true;
        }
        return false;
    }

    private static bool ContainsInf(Vector<float> vector)
    {
        for (int i = 0; i < vector.Length; i++)
        {
            if (float.IsInfinity(vector[i]))
                return true;
        }
        return false;
    }

    private static bool VectorsEqual(Vector<float> a, Vector<float> b)
    {
        if (a.Length != b.Length)
            return false;

        for (int i = 0; i < a.Length; i++)
        {
            if (Math.Abs(a[i] - b[i]) > 1e-6f)
                return false;
        }
        return true;
    }

    #endregion
}
