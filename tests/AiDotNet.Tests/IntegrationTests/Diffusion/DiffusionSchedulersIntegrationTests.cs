using System;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Diffusion;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Diffusion;

/// <summary>
/// Integration tests for Diffusion Schedulers - SchedulerConfig, DDIMScheduler, PNDMScheduler.
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
    public void DDIMScheduler_Construction_WithStableDiffusionConfig_Succeeds()
    {
        var config = SchedulerConfig<double>.CreateStableDiffusion();
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
    public void DDIMScheduler_Float_Construction_Succeeds()
    {
        var config = SchedulerConfig<float>.CreateDefault();
        var scheduler = new DDIMScheduler<float>(config);

        Assert.NotNull(scheduler);
    }

    [Fact]
    public void DDIMScheduler_Config_IsAccessible()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new DDIMScheduler<double>(config);

        Assert.Same(config, scheduler.Config);
    }

    #endregion

    #region PNDMScheduler Tests

    [Fact]
    public void PNDMScheduler_Construction_WithDefaultConfig_Succeeds()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new PNDMScheduler<double>(config);

        Assert.NotNull(scheduler);
    }

    [Fact]
    public void PNDMScheduler_Construction_WithStableDiffusionConfig_Succeeds()
    {
        var config = SchedulerConfig<double>.CreateStableDiffusion();
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
    public void PNDMScheduler_Float_Construction_Succeeds()
    {
        var config = SchedulerConfig<float>.CreateDefault();
        var scheduler = new PNDMScheduler<float>(config);

        Assert.NotNull(scheduler);
    }

    [Fact]
    public void PNDMScheduler_Config_IsAccessible()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new PNDMScheduler<double>(config);

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

        ddim.SetTimesteps(50);
        pndm.SetTimesteps(50);

        Assert.Equal(ddim.Timesteps.Length, pndm.Timesteps.Length);
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
}
