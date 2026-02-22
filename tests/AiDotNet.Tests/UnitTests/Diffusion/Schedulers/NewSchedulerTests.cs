using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Schedulers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion.Schedulers;

/// <summary>
/// Tests for all new scheduler implementations added in Issue #262.
/// Covers DDPM, Euler, EulerAncestral, DPM-Solver++, LCM, FlowMatching, and UniPC.
/// </summary>
public class NewSchedulerTests
{
    #region DDPM Scheduler Tests

    [Fact]
    public void DDPMScheduler_Constructor_CreatesSuccessfully()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new DDPMScheduler<double>(config);

        Assert.NotNull(scheduler);
        Assert.Equal(1000, scheduler.TrainTimesteps);
    }

    [Fact]
    public void DDPMScheduler_Step_ReturnsFiniteValues()
    {
        var config = new SchedulerConfig<double>(trainTimesteps: 50, betaStart: 0.0001, betaEnd: 0.02);
        var scheduler = new DDPMScheduler<double>(config);
        scheduler.SetTimesteps(10);
        int t = scheduler.Timesteps[0];

        var sample = new Vector<double>(new double[] { 0.1, -0.2, 0.3, -0.4 });
        var eps = new Vector<double>(new double[] { 0.05, 0.02, -0.01, -0.03 });

        var result = scheduler.Step(eps, t, sample, eta: 0.0);

        Assert.Equal(sample.Length, result.Length);
        foreach (var v in result)
        {
            Assert.False(double.IsNaN(v), "Result contains NaN");
            Assert.False(double.IsInfinity(v), "Result contains Infinity");
        }
    }

    [Fact]
    public void DDPMScheduler_FullLoop_ProducesFiniteResults()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new DDPMScheduler<double>(config);
        scheduler.SetTimesteps(20);

        var sample = CreateRandomVector(64, 42);

        foreach (var t in scheduler.Timesteps)
        {
            var modelOutput = CreateRandomVector(64, t);
            sample = scheduler.Step(modelOutput, t, sample, eta: 0.0);
        }

        AssertVectorIsFinite(sample);
    }

    #endregion

    #region Euler Discrete Scheduler Tests

    [Fact]
    public void EulerDiscreteScheduler_Constructor_CreatesSuccessfully()
    {
        var config = SchedulerConfig<double>.CreateStableDiffusion();
        var scheduler = new EulerDiscreteScheduler<double>(config);

        Assert.NotNull(scheduler);
        Assert.Equal(1000, scheduler.TrainTimesteps);
    }

    [Fact]
    public void EulerDiscreteScheduler_Step_ReturnsFiniteValues()
    {
        var config = new SchedulerConfig<double>(trainTimesteps: 50, betaStart: 0.0001, betaEnd: 0.02);
        var scheduler = new EulerDiscreteScheduler<double>(config);
        scheduler.SetTimesteps(10);
        int t = scheduler.Timesteps[0];

        var sample = new Vector<double>(new double[] { 0.1, -0.2, 0.3, -0.4 });
        var eps = new Vector<double>(new double[] { 0.05, 0.02, -0.01, -0.03 });

        var result = scheduler.Step(eps, t, sample, eta: 0.0);

        Assert.Equal(sample.Length, result.Length);
        AssertVectorIsFinite(result);
    }

    [Fact]
    public void EulerDiscreteScheduler_IsDeterministic()
    {
        var config = new SchedulerConfig<double>(trainTimesteps: 100, betaStart: 0.0001, betaEnd: 0.02);
        var scheduler = new EulerDiscreteScheduler<double>(config);
        scheduler.SetTimesteps(20);
        int t = scheduler.Timesteps[0];

        var sample = new Vector<double>(new double[] { 0.1, -0.2, 0.3 });
        var eps = new Vector<double>(new double[] { 0.05, 0.02, -0.01 });

        var result1 = scheduler.Step(eps, t, sample, eta: 0.0);
        var result2 = scheduler.Step(eps, t, sample, eta: 0.0);

        for (int i = 0; i < result1.Length; i++)
        {
            Assert.Equal(result1[i], result2[i], precision: 12);
        }
    }

    #endregion

    #region Euler Ancestral Discrete Scheduler Tests

    [Fact]
    public void EulerAncestralScheduler_Constructor_CreatesSuccessfully()
    {
        var config = SchedulerConfig<double>.CreateStableDiffusion();
        var scheduler = new EulerAncestralDiscreteScheduler<double>(config);

        Assert.NotNull(scheduler);
    }

    [Fact]
    public void EulerAncestralScheduler_Step_ReturnsFiniteValues()
    {
        var config = new SchedulerConfig<double>(trainTimesteps: 50, betaStart: 0.0001, betaEnd: 0.02);
        var scheduler = new EulerAncestralDiscreteScheduler<double>(config);
        scheduler.SetTimesteps(10);
        int t = scheduler.Timesteps[0];

        var sample = new Vector<double>(new double[] { 0.1, -0.2, 0.3, -0.4 });
        var eps = new Vector<double>(new double[] { 0.05, 0.02, -0.01, -0.03 });

        var result = scheduler.Step(eps, t, sample, eta: 0.0);

        Assert.Equal(sample.Length, result.Length);
        AssertVectorIsFinite(result);
    }

    [Fact]
    public void EulerAncestralScheduler_WithNoise_DiffersFromDeterministic()
    {
        var config = new SchedulerConfig<double>(trainTimesteps: 100, betaStart: 0.0001, betaEnd: 0.02);
        var scheduler = new EulerAncestralDiscreteScheduler<double>(config);
        scheduler.SetTimesteps(20);
        int t = scheduler.Timesteps[0];

        var sample = new Vector<double>(new double[] { 0.2, 0.0, -0.1 });
        var eps = new Vector<double>(new double[] { 0.01, -0.02, 0.03 });
        var noise = new Vector<double>(new double[] { 0.3, -0.5, 0.7 });

        var detResult = scheduler.Step(eps, t, sample, eta: 0.0);
        var stochResult = scheduler.Step(eps, t, sample, eta: 1.0, noise: noise);

        bool anyDiff = false;
        for (int i = 0; i < sample.Length; i++)
        {
            if (Math.Abs(detResult[i] - stochResult[i]) > 1e-9)
            {
                anyDiff = true;
                break;
            }
        }
        Assert.True(anyDiff, "Stochastic step should differ from deterministic");
    }

    #endregion

    #region DPM-Solver++ Multistep Scheduler Tests

    [Fact]
    public void DPMSolverMultistep_Constructor_CreatesSuccessfully()
    {
        var config = SchedulerConfig<double>.CreateStableDiffusion();
        var scheduler = new DPMSolverMultistepScheduler<double>(config, solverOrder: 2);

        Assert.NotNull(scheduler);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(4)]
    [InlineData(-1)]
    public void DPMSolverMultistep_InvalidOrder_ThrowsException(int invalidOrder)
    {
        var config = SchedulerConfig<double>.CreateDefault();
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new DPMSolverMultistepScheduler<double>(config, solverOrder: invalidOrder));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void DPMSolverMultistep_ValidOrders_CreateSuccessfully(int order)
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new DPMSolverMultistepScheduler<double>(config, solverOrder: order);
        Assert.NotNull(scheduler);
    }

    [Fact]
    public void DPMSolverMultistep_FullLoop_ProducesFiniteResults()
    {
        var config = SchedulerConfig<double>.CreateStableDiffusion();
        var scheduler = new DPMSolverMultistepScheduler<double>(config, solverOrder: 2);
        scheduler.SetTimesteps(20);

        var sample = CreateRandomVector(32, 42);

        foreach (var t in scheduler.Timesteps)
        {
            var modelOutput = CreateRandomVector(32, t);
            sample = scheduler.Step(modelOutput, t, sample, eta: 0.0);
        }

        AssertVectorIsFinite(sample);
    }

    #endregion

    #region LCM Scheduler Tests

    [Fact]
    public void LCMScheduler_Constructor_CreatesSuccessfully()
    {
        var config = SchedulerConfig<double>.CreateLCM();
        var scheduler = new LCMScheduler<double>(config);

        Assert.NotNull(scheduler);
    }

    [Fact]
    public void LCMScheduler_Step_ReturnsFiniteValues()
    {
        var config = SchedulerConfig<double>.CreateLCM();
        var scheduler = new LCMScheduler<double>(config);
        scheduler.SetTimesteps(4); // LCM uses very few steps

        var sample = new Vector<double>(new double[] { 0.1, -0.2, 0.3, -0.4 });
        var eps = new Vector<double>(new double[] { 0.05, 0.02, -0.01, -0.03 });

        var result = scheduler.Step(eps, scheduler.Timesteps[0], sample, eta: 0.0);

        Assert.Equal(sample.Length, result.Length);
        AssertVectorIsFinite(result);
    }

    [Fact]
    public void LCMScheduler_FullLoop_FewSteps_ProducesFiniteResults()
    {
        var config = SchedulerConfig<double>.CreateLCM();
        var scheduler = new LCMScheduler<double>(config);
        scheduler.SetTimesteps(4); // Typical LCM: 2-8 steps

        var sample = CreateRandomVector(16, 42);

        foreach (var t in scheduler.Timesteps)
        {
            var modelOutput = CreateRandomVector(16, t);
            sample = scheduler.Step(modelOutput, t, sample, eta: 0.0);
        }

        AssertVectorIsFinite(sample);
    }

    #endregion

    #region Flow Matching Scheduler Tests

    [Fact]
    public void FlowMatchingScheduler_Constructor_CreatesSuccessfully()
    {
        var config = SchedulerConfig<double>.CreateRectifiedFlow();
        var scheduler = new FlowMatchingScheduler<double>(config);

        Assert.NotNull(scheduler);
        Assert.Equal(1000, scheduler.TrainTimesteps);
    }

    [Fact]
    public void FlowMatchingScheduler_CreateDefault_Works()
    {
        var scheduler = FlowMatchingScheduler<double>.CreateDefault();
        Assert.NotNull(scheduler);
    }

    [Fact]
    public void FlowMatchingScheduler_AddNoise_UsesLinearInterpolation()
    {
        var config = SchedulerConfig<double>.CreateRectifiedFlow();
        var scheduler = new FlowMatchingScheduler<double>(config);
        scheduler.SetTimesteps(20);

        var original = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var noise = new Vector<double>(new double[] { 0.5, 0.5, 0.5 });

        // At timestep 0, t=0, should return mostly original: (1-0)*original + 0*noise = original
        var resultLow = scheduler.AddNoise(original, noise, timestep: 0);
        for (int i = 0; i < original.Length; i++)
        {
            Assert.InRange(resultLow[i], original[i] * 0.95, original[i] * 1.05 + 0.1);
        }

        // At high timestep (near max), should return mostly noise
        var resultHigh = scheduler.AddNoise(original, noise, timestep: 999);

        // Higher timestep should mean more noise contribution
        double sumDiffLow = 0, sumDiffHigh = 0;
        for (int i = 0; i < original.Length; i++)
        {
            sumDiffLow += Math.Abs(resultLow[i] - original[i]);
            sumDiffHigh += Math.Abs(resultHigh[i] - original[i]);
        }
        Assert.True(sumDiffHigh > sumDiffLow,
            "Higher timestep should result in more deviation from original");
    }

    [Fact]
    public void FlowMatchingScheduler_Step_ReturnsFiniteValues()
    {
        var config = SchedulerConfig<double>.CreateRectifiedFlow();
        var scheduler = new FlowMatchingScheduler<double>(config);
        scheduler.SetTimesteps(28); // SD3 recommended

        var sample = new Vector<double>(new double[] { 0.1, -0.2, 0.3, -0.4 });
        var velocity = new Vector<double>(new double[] { 0.05, 0.02, -0.01, -0.03 });

        var result = scheduler.Step(velocity, scheduler.Timesteps[0], sample, eta: 0.0);

        Assert.Equal(sample.Length, result.Length);
        AssertVectorIsFinite(result);
    }

    [Fact]
    public void FlowMatchingScheduler_FullLoop_ProducesFiniteResults()
    {
        var config = SchedulerConfig<double>.CreateRectifiedFlow();
        var scheduler = new FlowMatchingScheduler<double>(config);
        scheduler.SetTimesteps(20);

        var sample = CreateRandomVector(32, 42);

        foreach (var t in scheduler.Timesteps)
        {
            var modelOutput = CreateRandomVector(32, t);
            sample = scheduler.Step(modelOutput, t, sample, eta: 0.0);
        }

        AssertVectorIsFinite(sample);
    }

    [Fact]
    public void FlowMatchingScheduler_GetState_ContainsSchedulerType()
    {
        var scheduler = FlowMatchingScheduler<double>.CreateDefault();
        scheduler.SetTimesteps(10);

        var state = scheduler.GetState();

        Assert.True(state.ContainsKey("scheduler_type"));
        Assert.Equal("FlowMatching", state["scheduler_type"]);
    }

    #endregion

    #region UniPC Scheduler Tests

    [Fact]
    public void UniPCScheduler_Constructor_CreatesSuccessfully()
    {
        var config = SchedulerConfig<double>.CreateStableDiffusion();
        var scheduler = new UniPCScheduler<double>(config);

        Assert.NotNull(scheduler);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(4)]
    [InlineData(-1)]
    public void UniPCScheduler_InvalidOrder_ThrowsException(int invalidOrder)
    {
        var config = SchedulerConfig<double>.CreateDefault();
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new UniPCScheduler<double>(config, solverOrder: invalidOrder));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void UniPCScheduler_ValidOrders_CreateSuccessfully(int order)
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new UniPCScheduler<double>(config, solverOrder: order);
        Assert.NotNull(scheduler);
    }

    [Fact]
    public void UniPCScheduler_Step_ReturnsFiniteValues()
    {
        var config = new SchedulerConfig<double>(trainTimesteps: 50, betaStart: 0.0001, betaEnd: 0.02);
        var scheduler = new UniPCScheduler<double>(config, solverOrder: 2, useCorrectorStep: true);
        scheduler.SetTimesteps(10);
        int t = scheduler.Timesteps[0];

        var sample = new Vector<double>(new double[] { 0.1, -0.2, 0.3, -0.4 });
        var eps = new Vector<double>(new double[] { 0.05, 0.02, -0.01, -0.03 });

        var result = scheduler.Step(eps, t, sample, eta: 0.0);

        Assert.Equal(sample.Length, result.Length);
        AssertVectorIsFinite(result);
    }

    [Fact]
    public void UniPCScheduler_WithoutCorrectorStep_FullLoop()
    {
        var config = SchedulerConfig<double>.CreateStableDiffusion();
        var scheduler = new UniPCScheduler<double>(config, solverOrder: 2, useCorrectorStep: false);
        scheduler.SetTimesteps(15);

        var sample = CreateRandomVector(32, 42);

        foreach (var t in scheduler.Timesteps)
        {
            var modelOutput = CreateRandomVector(32, t);
            sample = scheduler.Step(modelOutput, t, sample, eta: 0.0);
        }

        AssertVectorIsFinite(sample);
    }

    [Fact]
    public void UniPCScheduler_WithCorrectorStep_FullLoop()
    {
        var config = SchedulerConfig<double>.CreateStableDiffusion();
        var scheduler = new UniPCScheduler<double>(config, solverOrder: 2, useCorrectorStep: true);
        scheduler.SetTimesteps(15);

        var sample = CreateRandomVector(32, 42);

        foreach (var t in scheduler.Timesteps)
        {
            var modelOutput = CreateRandomVector(32, t);
            sample = scheduler.Step(modelOutput, t, sample, eta: 0.0);
        }

        AssertVectorIsFinite(sample);
    }

    [Fact]
    public void UniPCScheduler_GetState_ContainsSchedulerType()
    {
        var config = SchedulerConfig<double>.CreateStableDiffusion();
        var scheduler = new UniPCScheduler<double>(config, solverOrder: 2, useCorrectorStep: true);
        scheduler.SetTimesteps(10);

        var state = scheduler.GetState();

        Assert.True(state.ContainsKey("scheduler_type"));
        Assert.Equal("UniPC", state["scheduler_type"]);
        Assert.Equal(2, state["solver_order"]);
        Assert.Equal(true, state["use_corrector"]);
    }

    #endregion

    #region SchedulerConfig Factory Method Tests

    [Fact]
    public void SchedulerConfig_CreateRectifiedFlow_HasCorrectDefaults()
    {
        var config = SchedulerConfig<double>.CreateRectifiedFlow();

        Assert.Equal(1000, config.TrainTimesteps);
        Assert.Equal(BetaSchedule.Linear, config.BetaSchedule);
        Assert.Equal(DiffusionPredictionType.VPrediction, config.PredictionType);
        Assert.False(config.ClipSample);
    }

    [Fact]
    public void SchedulerConfig_CreateLCM_HasCorrectDefaults()
    {
        var config = SchedulerConfig<double>.CreateLCM();

        Assert.Equal(1000, config.TrainTimesteps);
        Assert.Equal(BetaSchedule.ScaledLinear, config.BetaSchedule);
        Assert.Equal(DiffusionPredictionType.Epsilon, config.PredictionType);
    }

    #endregion

    #region Cross-Scheduler Contract Tests

    [Fact]
    public void AllSchedulers_SetTimesteps_ProducesDescendingSequence()
    {
        var sdConfig = SchedulerConfig<double>.CreateStableDiffusion();
        var defaultConfig = SchedulerConfig<double>.CreateDefault();
        var flowConfig = SchedulerConfig<double>.CreateRectifiedFlow();

        var schedulers = new INoiseScheduler<double>[]
        {
            new DDPMScheduler<double>(defaultConfig),
            new EulerDiscreteScheduler<double>(sdConfig),
            new EulerAncestralDiscreteScheduler<double>(sdConfig),
            new DPMSolverMultistepScheduler<double>(sdConfig),
            new LCMScheduler<double>(SchedulerConfig<double>.CreateLCM()),
            new FlowMatchingScheduler<double>(flowConfig),
            new UniPCScheduler<double>(sdConfig),
        };

        foreach (var scheduler in schedulers)
        {
            scheduler.SetTimesteps(20);
            var timesteps = scheduler.Timesteps;

            Assert.True(timesteps.Length > 0,
                $"{scheduler.GetType().Name} should produce non-empty timesteps");

            for (int i = 1; i < timesteps.Length; i++)
            {
                Assert.True(timesteps[i] < timesteps[i - 1],
                    $"{scheduler.GetType().Name}: Timestep[{i}] ({timesteps[i]}) should be < Timestep[{i - 1}] ({timesteps[i - 1]})");
            }
        }
    }

    [Fact]
    public void AllSchedulers_Step_ReturnsCorrectLength()
    {
        var sdConfig = SchedulerConfig<double>.CreateStableDiffusion();
        var defaultConfig = SchedulerConfig<double>.CreateDefault();
        var flowConfig = SchedulerConfig<double>.CreateRectifiedFlow();

        var schedulers = new INoiseScheduler<double>[]
        {
            new DDPMScheduler<double>(defaultConfig),
            new EulerDiscreteScheduler<double>(sdConfig),
            new EulerAncestralDiscreteScheduler<double>(sdConfig),
            new DPMSolverMultistepScheduler<double>(sdConfig),
            new LCMScheduler<double>(SchedulerConfig<double>.CreateLCM()),
            new FlowMatchingScheduler<double>(flowConfig),
            new UniPCScheduler<double>(sdConfig),
        };

        var sample = new Vector<double>(new double[] { 0.1, -0.2, 0.3, -0.4 });
        var modelOutput = new Vector<double>(new double[] { 0.05, 0.02, -0.01, -0.03 });

        foreach (var scheduler in schedulers)
        {
            scheduler.SetTimesteps(10);
            int t = scheduler.Timesteps[0];

            var result = scheduler.Step(modelOutput, t, sample, eta: 0.0);

            Assert.Equal(sample.Length, result.Length);
            AssertVectorIsFinite(result);
        }
    }

    [Fact]
    public void AllSchedulers_NullModelOutput_ThrowsArgumentNullException()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var schedulers = new INoiseScheduler<double>[]
        {
            new DDPMScheduler<double>(config),
            new EulerDiscreteScheduler<double>(config),
            new EulerAncestralDiscreteScheduler<double>(config),
            new DPMSolverMultistepScheduler<double>(config),
            new FlowMatchingScheduler<double>(SchedulerConfig<double>.CreateRectifiedFlow()),
            new UniPCScheduler<double>(config),
        };

        var sample = new Vector<double>(new double[] { 0.1, 0.2 });

        foreach (var scheduler in schedulers)
        {
            scheduler.SetTimesteps(10);
            var t = scheduler.Timesteps[0];
            var s = scheduler;
            Assert.Throws<ArgumentNullException>(() =>
            {
                s.Step(null!, t, sample, 0.0);
            });
        }
    }

    #endregion

    #region Helpers

    private static Vector<double> CreateRandomVector(int length, int seed)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        var data = new double[length];
        for (int i = 0; i < length; i++)
        {
            data[i] = random.NextDouble() * 2 - 1;
        }
        return new Vector<double>(data);
    }

    private static void AssertVectorIsFinite(Vector<double> vector)
    {
        for (int i = 0; i < vector.Length; i++)
        {
            Assert.False(double.IsNaN(vector[i]), $"Vector[{i}] is NaN");
            Assert.False(double.IsInfinity(vector[i]), $"Vector[{i}] is Infinity");
        }
    }

    #endregion
}
