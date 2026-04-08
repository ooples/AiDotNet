using AiDotNet.Enums;
using AiDotNet.Diffusion.Schedulers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Diffusion;

/// <summary>
/// Integration tests for Diffusion Schedulers - SchedulerConfig, all 15 NoiseScheduler implementations.
/// </summary>
public class DiffusionSchedulersIntegrationTests
{
    #region SchedulerConfig Tests

    [Fact]
    public void SchedulerConfig_CreateDefault_ReturnsValidConfig()
    {
        var config = SchedulerConfig<double>.CreateDefault();

        Assert.NotNull(config);
        Assert.Equal(1000, config.TrainTimesteps);
        Assert.Equal(BetaSchedule.Linear, config.BetaSchedule);
        Assert.Equal(DiffusionPredictionType.Epsilon, config.PredictionType);
        Assert.False(config.ClipSample);
    }

    [Fact]
    public void SchedulerConfig_CreateStableDiffusion_ReturnsValidConfig()
    {
        var config = SchedulerConfig<double>.CreateStableDiffusion();

        Assert.NotNull(config);
        Assert.Equal(1000, config.TrainTimesteps);
        Assert.Equal(BetaSchedule.ScaledLinear, config.BetaSchedule);
        Assert.Equal(DiffusionPredictionType.Epsilon, config.PredictionType);
    }

    [Fact]
    public void SchedulerConfig_CreateRectifiedFlow_ReturnsValidConfig()
    {
        var config = SchedulerConfig<double>.CreateRectifiedFlow();

        Assert.NotNull(config);
        Assert.Equal(1000, config.TrainTimesteps);
        Assert.Equal(BetaSchedule.Linear, config.BetaSchedule);
        Assert.Equal(DiffusionPredictionType.VPrediction, config.PredictionType);
    }

    [Fact]
    public void SchedulerConfig_CreateLCM_ReturnsValidConfig()
    {
        var config = SchedulerConfig<double>.CreateLCM();

        Assert.NotNull(config);
        Assert.Equal(1000, config.TrainTimesteps);
        Assert.Equal(BetaSchedule.ScaledLinear, config.BetaSchedule);
        Assert.Equal(DiffusionPredictionType.Epsilon, config.PredictionType);
    }

    [Fact]
    public void SchedulerConfig_CustomParameters_SetsCorrectly()
    {
        var config = new SchedulerConfig<double>(
            trainTimesteps: 500,
            betaStart: 0.001,
            betaEnd: 0.05,
            betaSchedule: BetaSchedule.SquaredCosine,
            clipSample: true,
            predictionType: DiffusionPredictionType.Sample);

        Assert.Equal(500, config.TrainTimesteps);
        Assert.Equal(0.001, config.BetaStart);
        Assert.Equal(0.05, config.BetaEnd);
        Assert.Equal(BetaSchedule.SquaredCosine, config.BetaSchedule);
        Assert.True(config.ClipSample);
        Assert.Equal(DiffusionPredictionType.Sample, config.PredictionType);
    }

    [Fact]
    public void SchedulerConfig_InvalidTimesteps_ThrowsException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new SchedulerConfig<double>(trainTimesteps: 1, betaStart: 0.0001, betaEnd: 0.02));
    }

    [Fact]
    public void SchedulerConfig_Float_CreateDefault_Works()
    {
        var config = SchedulerConfig<float>.CreateDefault();

        Assert.NotNull(config);
        Assert.Equal(1000, config.TrainTimesteps);
    }

    [Fact]
    public void SchedulerConfig_AllBetaSchedules_Work()
    {
        var schedules = new[] { BetaSchedule.Linear, BetaSchedule.ScaledLinear, BetaSchedule.SquaredCosine };

        foreach (var schedule in schedules)
        {
            var config = new SchedulerConfig<double>(
                trainTimesteps: 100,
                betaStart: 0.0001,
                betaEnd: 0.02,
                betaSchedule: schedule);

            Assert.Equal(schedule, config.BetaSchedule);
        }
    }

    [Fact]
    public void SchedulerConfig_AllPredictionTypes_Work()
    {
        var predictions = new[] { DiffusionPredictionType.Epsilon, DiffusionPredictionType.Sample, DiffusionPredictionType.VPrediction };

        foreach (var predType in predictions)
        {
            var config = new SchedulerConfig<double>(
                trainTimesteps: 100,
                betaStart: 0.0001,
                betaEnd: 0.02,
                predictionType: predType);

            Assert.Equal(predType, config.PredictionType);
        }
    }

    #endregion

    #region DDIMScheduler Tests

    [Fact]
    public void DDIMScheduler_Construction_WithDefaultConfig_Succeeds()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new DDIMScheduler<double>(config);
        Assert.NotNull(scheduler);
    }

    [Fact]
    public void DDIMScheduler_SetTimesteps_Succeeds()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new DDIMScheduler<double>(config);
        scheduler.SetTimesteps(50);
        Assert.Equal(50, scheduler.Timesteps.Length);
    }

    [Fact]
    public void DDIMScheduler_SetTimesteps_DifferentValues_Work()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new DDIMScheduler<double>(config);

        scheduler.SetTimesteps(20);
        Assert.Equal(20, scheduler.Timesteps.Length);

        scheduler.SetTimesteps(100);
        Assert.Equal(100, scheduler.Timesteps.Length);
    }

    [Fact]
    public void DDIMScheduler_Config_IsAccessible()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new DDIMScheduler<double>(config);
        Assert.Same(config, scheduler.Config);
    }

    [Fact]
    public void DDIMScheduler_TrainTimesteps_MatchesConfig()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new DDIMScheduler<double>(config);
        Assert.Equal(1000, scheduler.TrainTimesteps);
    }

    [Fact]
    public void DDIMScheduler_Timesteps_AreDescending()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new DDIMScheduler<double>(config);
        scheduler.SetTimesteps(20);

        for (int i = 0; i < scheduler.Timesteps.Length - 1; i++)
        {
            Assert.True(scheduler.Timesteps[i] > scheduler.Timesteps[i + 1],
                $"Timestep {i} ({scheduler.Timesteps[i]}) should be > timestep {i + 1} ({scheduler.Timesteps[i + 1]})");
        }
    }

    #endregion

    #region PNDMScheduler Tests

    [Fact]
    public void PNDMScheduler_Construction_Succeeds()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new PNDMScheduler<double>(config);
        Assert.NotNull(scheduler);
    }

    [Fact]
    public void PNDMScheduler_SetTimesteps_Succeeds()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new PNDMScheduler<double>(config);
        scheduler.SetTimesteps(50);
        Assert.Equal(50, scheduler.Timesteps.Length);
    }

    [Fact]
    public void PNDMScheduler_Config_IsAccessible()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new PNDMScheduler<double>(config);
        Assert.Same(config, scheduler.Config);
    }

    #endregion

    #region DDPMScheduler Tests

    [Fact]
    public void DDPMScheduler_Construction_WithDefaultConfig_Succeeds()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new DDPMScheduler<double>(config);
        Assert.NotNull(scheduler);
    }

    [Fact]
    public void DDPMScheduler_Construction_WithStableDiffusionConfig_Succeeds()
    {
        var config = SchedulerConfig<double>.CreateStableDiffusion();
        var scheduler = new DDPMScheduler<double>(config);
        Assert.NotNull(scheduler);
    }

    [Fact]
    public void DDPMScheduler_SetTimesteps_Succeeds()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new DDPMScheduler<double>(config);
        scheduler.SetTimesteps(50);
        Assert.Equal(50, scheduler.Timesteps.Length);
    }

    [Fact]
    public void DDPMScheduler_Config_IsAccessible()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new DDPMScheduler<double>(config);
        Assert.Same(config, scheduler.Config);
        Assert.Equal(1000, scheduler.TrainTimesteps);
    }

    [Fact]
    public void DDPMScheduler_Float_Works()
    {
        var config = SchedulerConfig<float>.CreateDefault();
        var scheduler = new DDPMScheduler<float>(config);
        scheduler.SetTimesteps(20);
        Assert.Equal(20, scheduler.Timesteps.Length);
    }

    #endregion

    #region EulerDiscreteScheduler Tests

    [Fact]
    public void EulerDiscreteScheduler_Construction_Succeeds()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new EulerDiscreteScheduler<double>(config);
        Assert.NotNull(scheduler);
    }

    [Fact]
    public void EulerDiscreteScheduler_WithStableDiffusionConfig_Succeeds()
    {
        var config = SchedulerConfig<double>.CreateStableDiffusion();
        var scheduler = new EulerDiscreteScheduler<double>(config);
        Assert.NotNull(scheduler);
    }

    [Fact]
    public void EulerDiscreteScheduler_SetTimesteps_Succeeds()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new EulerDiscreteScheduler<double>(config);
        scheduler.SetTimesteps(30);
        Assert.Equal(30, scheduler.Timesteps.Length);
    }

    [Fact]
    public void EulerDiscreteScheduler_Timesteps_AreDescending()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new EulerDiscreteScheduler<double>(config);
        scheduler.SetTimesteps(20);

        for (int i = 0; i < scheduler.Timesteps.Length - 1; i++)
        {
            Assert.True(scheduler.Timesteps[i] > scheduler.Timesteps[i + 1]);
        }
    }

    #endregion

    #region EulerAncestralDiscreteScheduler Tests

    [Fact]
    public void EulerAncestralDiscreteScheduler_Construction_Succeeds()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new EulerAncestralDiscreteScheduler<double>(config);
        Assert.NotNull(scheduler);
    }

    [Fact]
    public void EulerAncestralDiscreteScheduler_SetTimesteps_Succeeds()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new EulerAncestralDiscreteScheduler<double>(config);
        scheduler.SetTimesteps(25);
        Assert.Equal(25, scheduler.Timesteps.Length);
    }

    [Fact]
    public void EulerAncestralDiscreteScheduler_Config_IsAccessible()
    {
        var config = SchedulerConfig<double>.CreateStableDiffusion();
        var scheduler = new EulerAncestralDiscreteScheduler<double>(config);
        Assert.Same(config, scheduler.Config);
    }

    #endregion

    #region HeunDiscreteScheduler Tests

    [Fact]
    public void HeunDiscreteScheduler_Construction_Succeeds()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new HeunDiscreteScheduler<double>(config);
        Assert.NotNull(scheduler);
    }

    [Fact]
    public void HeunDiscreteScheduler_SetTimesteps_Succeeds()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new HeunDiscreteScheduler<double>(config);
        scheduler.SetTimesteps(30);
        Assert.Equal(30, scheduler.Timesteps.Length);
    }

    [Fact]
    public void HeunDiscreteScheduler_WithSquaredCosineSchedule_Succeeds()
    {
        var config = new SchedulerConfig<double>(
            trainTimesteps: 1000,
            betaStart: 0.0001,
            betaEnd: 0.02,
            betaSchedule: BetaSchedule.SquaredCosine);
        var scheduler = new HeunDiscreteScheduler<double>(config);
        scheduler.SetTimesteps(20);
        Assert.Equal(20, scheduler.Timesteps.Length);
    }

    #endregion

    #region DPMSolverMultistepScheduler Tests

    [Fact]
    public void DPMSolverMultistepScheduler_Construction_Succeeds()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new DPMSolverMultistepScheduler<double>(config);
        Assert.NotNull(scheduler);
    }

    [Fact]
    public void DPMSolverMultistepScheduler_WithStableDiffusionConfig_Succeeds()
    {
        var config = SchedulerConfig<double>.CreateStableDiffusion();
        var scheduler = new DPMSolverMultistepScheduler<double>(config);
        Assert.NotNull(scheduler);
    }

    [Fact]
    public void DPMSolverMultistepScheduler_SetTimesteps_Succeeds()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new DPMSolverMultistepScheduler<double>(config);
        scheduler.SetTimesteps(20);
        Assert.Equal(20, scheduler.Timesteps.Length);
    }

    [Fact]
    public void DPMSolverMultistepScheduler_Config_IsAccessible()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new DPMSolverMultistepScheduler<double>(config);
        Assert.Same(config, scheduler.Config);
    }

    #endregion

    #region DPMSolverSinglestepScheduler Tests

    [Fact]
    public void DPMSolverSinglestepScheduler_Construction_Succeeds()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new DPMSolverSinglestepScheduler<double>(config);
        Assert.NotNull(scheduler);
    }

    [Fact]
    public void DPMSolverSinglestepScheduler_SetTimesteps_Succeeds()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new DPMSolverSinglestepScheduler<double>(config);
        scheduler.SetTimesteps(25);
        Assert.Equal(25, scheduler.Timesteps.Length);
    }

    #endregion

    #region DPMSolverSDEScheduler Tests

    [Fact]
    public void DPMSolverSDEScheduler_Construction_Succeeds()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new DPMSolverSDEScheduler<double>(config);
        Assert.NotNull(scheduler);
    }

    [Fact]
    public void DPMSolverSDEScheduler_SetTimesteps_Succeeds()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new DPMSolverSDEScheduler<double>(config);
        scheduler.SetTimesteps(30);
        Assert.Equal(30, scheduler.Timesteps.Length);
    }

    #endregion

    #region DEISMultistepScheduler Tests

    [Fact]
    public void DEISMultistepScheduler_Construction_Succeeds()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new DEISMultistepScheduler<double>(config);
        Assert.NotNull(scheduler);
    }

    [Fact]
    public void DEISMultistepScheduler_SetTimesteps_Succeeds()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new DEISMultistepScheduler<double>(config);
        scheduler.SetTimesteps(20);
        Assert.Equal(20, scheduler.Timesteps.Length);
    }

    [Fact]
    public void DEISMultistepScheduler_WithStableDiffusionConfig_Succeeds()
    {
        var config = SchedulerConfig<double>.CreateStableDiffusion();
        var scheduler = new DEISMultistepScheduler<double>(config);
        scheduler.SetTimesteps(15);
        Assert.Equal(15, scheduler.Timesteps.Length);
    }

    #endregion

    #region LMSDiscreteScheduler Tests

    [Fact]
    public void LMSDiscreteScheduler_Construction_Succeeds()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new LMSDiscreteScheduler<double>(config);
        Assert.NotNull(scheduler);
    }

    [Fact]
    public void LMSDiscreteScheduler_SetTimesteps_Succeeds()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new LMSDiscreteScheduler<double>(config);
        scheduler.SetTimesteps(30);
        Assert.Equal(30, scheduler.Timesteps.Length);
    }

    #endregion

    #region UniPCScheduler Tests

    [Fact]
    public void UniPCScheduler_Construction_Succeeds()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new UniPCScheduler<double>(config);
        Assert.NotNull(scheduler);
    }

    [Fact]
    public void UniPCScheduler_SetTimesteps_Succeeds()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new UniPCScheduler<double>(config);
        scheduler.SetTimesteps(20);
        Assert.Equal(20, scheduler.Timesteps.Length);
    }

    [Fact]
    public void UniPCScheduler_WithStableDiffusionConfig_Succeeds()
    {
        var config = SchedulerConfig<double>.CreateStableDiffusion();
        var scheduler = new UniPCScheduler<double>(config);
        scheduler.SetTimesteps(25);
        Assert.Equal(25, scheduler.Timesteps.Length);
    }

    #endregion

    #region ConsistencyModelScheduler Tests

    [Fact]
    public void ConsistencyModelScheduler_Construction_Succeeds()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new ConsistencyModelScheduler<double>(config);
        Assert.NotNull(scheduler);
    }

    [Fact]
    public void ConsistencyModelScheduler_SetTimesteps_Succeeds()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new ConsistencyModelScheduler<double>(config);
        scheduler.SetTimesteps(4);
        Assert.Equal(4, scheduler.Timesteps.Length);
    }

    [Fact]
    public void ConsistencyModelScheduler_FewSteps_Work()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new ConsistencyModelScheduler<double>(config);
        // Consistency models are designed for very few steps
        scheduler.SetTimesteps(2);
        Assert.Equal(2, scheduler.Timesteps.Length);
    }

    #endregion

    #region FlowMatchingScheduler Tests

    [Fact]
    public void FlowMatchingScheduler_Construction_Succeeds()
    {
        var config = SchedulerConfig<double>.CreateRectifiedFlow();
        var scheduler = new FlowMatchingScheduler<double>(config);
        Assert.NotNull(scheduler);
    }

    [Fact]
    public void FlowMatchingScheduler_SetTimesteps_Succeeds()
    {
        var config = SchedulerConfig<double>.CreateRectifiedFlow();
        var scheduler = new FlowMatchingScheduler<double>(config);
        scheduler.SetTimesteps(28);
        Assert.Equal(28, scheduler.Timesteps.Length);
    }

    [Fact]
    public void FlowMatchingScheduler_WithDefaultConfig_Succeeds()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new FlowMatchingScheduler<double>(config);
        scheduler.SetTimesteps(20);
        Assert.Equal(20, scheduler.Timesteps.Length);
    }

    #endregion

    #region LCMScheduler Tests

    [Fact]
    public void LCMScheduler_Construction_Succeeds()
    {
        var config = SchedulerConfig<double>.CreateLCM();
        var scheduler = new LCMScheduler<double>(config);
        Assert.NotNull(scheduler);
    }

    [Fact]
    public void LCMScheduler_SetTimesteps_Succeeds()
    {
        var config = SchedulerConfig<double>.CreateLCM();
        var scheduler = new LCMScheduler<double>(config);
        scheduler.SetTimesteps(4);
        Assert.Equal(4, scheduler.Timesteps.Length);
    }

    [Fact]
    public void LCMScheduler_FewSteps_Work()
    {
        // LCM is designed for very few steps (1-8)
        var config = SchedulerConfig<double>.CreateLCM();
        var scheduler = new LCMScheduler<double>(config);
        scheduler.SetTimesteps(2);
        Assert.Equal(2, scheduler.Timesteps.Length);
    }

    [Fact]
    public void LCMScheduler_Config_IsAccessible()
    {
        var config = SchedulerConfig<double>.CreateLCM();
        var scheduler = new LCMScheduler<double>(config);
        Assert.Same(config, scheduler.Config);
    }

    #endregion

    #region Cross-Scheduler Tests

    [Fact]
    public void AllSchedulers_SameConfig_ProduceSameTimestepCount()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var ddim = new DDIMScheduler<double>(config);
        var pndm = new PNDMScheduler<double>(config);
        var ddpm = new DDPMScheduler<double>(config);
        var euler = new EulerDiscreteScheduler<double>(config);

        ddim.SetTimesteps(50);
        pndm.SetTimesteps(50);
        ddpm.SetTimesteps(50);
        euler.SetTimesteps(50);

        Assert.Equal(50, ddim.Timesteps.Length);
        Assert.Equal(50, pndm.Timesteps.Length);
        Assert.Equal(50, ddpm.Timesteps.Length);
        Assert.Equal(50, euler.Timesteps.Length);
    }

    [Fact]
    public void AllSchedulers_InvalidTimesteps_Throw()
    {
        var config = SchedulerConfig<double>.CreateDefault();

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new DDIMScheduler<double>(config).SetTimesteps(0));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new DDIMScheduler<double>(config).SetTimesteps(-1));
    }

    [Fact]
    public void AllSchedulers_ExceedingTrainTimesteps_Throws()
    {
        var config = new SchedulerConfig<double>(
            trainTimesteps: 100,
            betaStart: 0.0001,
            betaEnd: 0.02);

        var scheduler = new DDIMScheduler<double>(config);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            scheduler.SetTimesteps(200));
    }

    [Fact]
    public void AllSchedulers_NullConfig_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new DDIMScheduler<double>(null));
        Assert.Throws<ArgumentNullException>(() => new PNDMScheduler<double>(null));
        Assert.Throws<ArgumentNullException>(() => new DDPMScheduler<double>(null));
        Assert.Throws<ArgumentNullException>(() => new EulerDiscreteScheduler<double>(null));
        Assert.Throws<ArgumentNullException>(() => new EulerAncestralDiscreteScheduler<double>(null));
        Assert.Throws<ArgumentNullException>(() => new HeunDiscreteScheduler<double>(null));
        Assert.Throws<ArgumentNullException>(() => new DPMSolverMultistepScheduler<double>(null));
        Assert.Throws<ArgumentNullException>(() => new DPMSolverSinglestepScheduler<double>(null));
        Assert.Throws<ArgumentNullException>(() => new DPMSolverSDEScheduler<double>(null));
        Assert.Throws<ArgumentNullException>(() => new DEISMultistepScheduler<double>(null));
        Assert.Throws<ArgumentNullException>(() => new LMSDiscreteScheduler<double>(null));
        Assert.Throws<ArgumentNullException>(() => new UniPCScheduler<double>(null));
        Assert.Throws<ArgumentNullException>(() => new ConsistencyModelScheduler<double>(null));
        Assert.Throws<ArgumentNullException>(() => new FlowMatchingScheduler<double>(null));
        Assert.Throws<ArgumentNullException>(() => new LCMScheduler<double>(null));
    }

    [Fact]
    public void AllSchedulers_TrainTimesteps_MatchConfig()
    {
        var config = new SchedulerConfig<double>(
            trainTimesteps: 500,
            betaStart: 0.0001,
            betaEnd: 0.02);

        Assert.Equal(500, new DDIMScheduler<double>(config).TrainTimesteps);
        Assert.Equal(500, new PNDMScheduler<double>(config).TrainTimesteps);
        Assert.Equal(500, new DDPMScheduler<double>(config).TrainTimesteps);
        Assert.Equal(500, new EulerDiscreteScheduler<double>(config).TrainTimesteps);
        Assert.Equal(500, new HeunDiscreteScheduler<double>(config).TrainTimesteps);
        Assert.Equal(500, new DPMSolverMultistepScheduler<double>(config).TrainTimesteps);
    }

    [Fact]
    public void AllSchedulers_Float_Construction_Works()
    {
        var config = SchedulerConfig<float>.CreateDefault();

        Assert.NotNull(new DDIMScheduler<float>(config));
        Assert.NotNull(new PNDMScheduler<float>(config));
        Assert.NotNull(new DDPMScheduler<float>(config));
        Assert.NotNull(new EulerDiscreteScheduler<float>(config));
        Assert.NotNull(new EulerAncestralDiscreteScheduler<float>(config));
        Assert.NotNull(new HeunDiscreteScheduler<float>(config));
        Assert.NotNull(new DPMSolverMultistepScheduler<float>(config));
        Assert.NotNull(new DPMSolverSinglestepScheduler<float>(config));
        Assert.NotNull(new DPMSolverSDEScheduler<float>(config));
        Assert.NotNull(new DEISMultistepScheduler<float>(config));
        Assert.NotNull(new LMSDiscreteScheduler<float>(config));
        Assert.NotNull(new UniPCScheduler<float>(config));
        Assert.NotNull(new ConsistencyModelScheduler<float>(config));
        Assert.NotNull(new FlowMatchingScheduler<float>(config));
        Assert.NotNull(new LCMScheduler<float>(config));
    }

    [Fact]
    public void AllSchedulers_WithSquaredCosineBeta_Construct()
    {
        var config = new SchedulerConfig<double>(
            trainTimesteps: 1000,
            betaStart: 0.0001,
            betaEnd: 0.02,
            betaSchedule: BetaSchedule.SquaredCosine);

        Assert.NotNull(new DDIMScheduler<double>(config));
        Assert.NotNull(new DDPMScheduler<double>(config));
        Assert.NotNull(new EulerDiscreteScheduler<double>(config));
        Assert.NotNull(new DPMSolverMultistepScheduler<double>(config));
    }

    [Fact]
    public void AllSchedulers_CanSetTimestepsMultipleTimes()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new EulerDiscreteScheduler<double>(config);

        scheduler.SetTimesteps(10);
        Assert.Equal(10, scheduler.Timesteps.Length);

        scheduler.SetTimesteps(50);
        Assert.Equal(50, scheduler.Timesteps.Length);

        scheduler.SetTimesteps(5);
        Assert.Equal(5, scheduler.Timesteps.Length);
    }

    [Fact]
    public void Scheduler_TimestepsInValidRange()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new DDIMScheduler<double>(config);
        scheduler.SetTimesteps(20);

        foreach (var t in scheduler.Timesteps)
        {
            Assert.True(t >= 0, $"Timestep {t} should be non-negative");
            Assert.True(t < config.TrainTimesteps, $"Timestep {t} should be < {config.TrainTimesteps}");
        }
    }

    [Fact]
    public void Scheduler_TimestepsAreUnique()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new DDIMScheduler<double>(config);
        scheduler.SetTimesteps(50);

        var unique = scheduler.Timesteps.Distinct().Count();
        Assert.Equal(scheduler.Timesteps.Length, unique);
    }

    #endregion
}
