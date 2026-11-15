using System;
using System.Linq;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Diffusion.ThreeD;
using AiDotNet.Enums;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion.Schedulers;

/// <summary>
/// diffusers' <c>DDIMScheduler.set_timesteps</c> spacings (<c>leading</c> with <c>steps_offset</c>, <c>trailing</c>,
/// <c>linspace</c>) against hand lists, and <c>set_alpha_to_one</c> on the last step, as Wonder3D's
/// <c>scheduler_config.json</c> configures them.
/// </summary>
public class DDIMTimestepSpacingTests
{
    [Fact]
    public void Leading_WithStepsOffsetOne_GivesWonder3DsFiftySteps()
    {
        // step_ratio = 1000 // 50 = 20; (arange(50) · 20)[::-1] + 1 = 981, 961, …, 1.
        var expected = Enumerable.Range(0, 50).Select(i => 981 - 20 * i).ToArray();
        Assert.Equal(expected, NoiseSchedulerBase<double>.SpacedTimesteps(1000, 50, TimestepSpacing.Leading, 1));
        Assert.Equal(new[] { 9, 7, 5, 3, 1 }, NoiseSchedulerBase<double>.SpacedTimesteps(10, 5, TimestepSpacing.Leading, 1));
    }

    [Fact]
    public void Trailing_IsTheDefaultAndCountsDownFromTheLastTrainingStep()
    {
        // round(arange(1000, 0, −20)) − 1 = 999, 979, …, 19.
        var expected = Enumerable.Range(0, 50).Select(i => 999 - 20 * i).ToArray();
        Assert.Equal(expected, NoiseSchedulerBase<double>.SpacedTimesteps(1000, 50, TimestepSpacing.Trailing, 0));
        Assert.Equal(TimestepSpacing.Trailing, SchedulerConfig<double>.CreateStableDiffusion().TimestepSpacing);
        var scheduler = new DDIMScheduler<double>(SchedulerConfig<double>.CreateStableDiffusion());
        scheduler.SetTimesteps(50);
        Assert.Equal(expected, scheduler.Timesteps);
    }

    [Fact]
    public void Linspace_RoundsHalfToEven()
    {
        // linspace(0, 999, 5) = 0, 249.75, 499.5, 749.25, 999 -> round -> 0, 250, 500, 749, 999 (reversed).
        Assert.Equal(new[] { 999, 749, 500, 250, 0 }, NoiseSchedulerBase<double>.SpacedTimesteps(1000, 5, TimestepSpacing.Linspace, 0));
    }

    [Fact]
    public void Wonder3DSamplerConfig_IsLeadingOffsetOneWithoutAlphaToOne()
    {
        var config = Wonder3DModel<double>.SamplerConfig(new Wonder3DOptions());
        Assert.Equal(TimestepSpacing.Leading, config.TimestepSpacing);
        Assert.Equal(1, config.StepsOffset);
        Assert.False(config.SetAlphaToOne);
        Assert.Equal(1000, config.TrainTimesteps);
        Assert.Equal(BetaSchedule.ScaledLinear, config.BetaSchedule);
        var scheduler = new DDIMScheduler<double>(config);
        scheduler.SetTimesteps(50);
        Assert.Equal(981, scheduler.Timesteps[0]);
        Assert.Equal(1, scheduler.Timesteps[49]);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void LastStep_UsesTheFinalAlphaCumprod(bool setAlphaToOne)
    {
        // Linear betas 0.1 … 0.5 over 10 steps: ᾱ₀ = 0.9, ᾱ₁ = 0.9 · (1 − (0.1 + 0.4/9)).
        var config = new SchedulerConfig<double>(10, 0.1, 0.5, BetaSchedule.Linear)
        {
            TimestepSpacing = TimestepSpacing.Leading, StepsOffset = 1, SetAlphaToOne = setAlphaToOne,
        };
        var scheduler = new DDIMScheduler<double>(config);
        scheduler.SetTimesteps(5);
        Assert.Equal(new[] { 9, 7, 5, 3, 1 }, scheduler.Timesteps);
        double a0 = 0.9, a1 = 0.9 * (1 - (0.1 + 0.4 / 9));
        Assert.Equal(a1, scheduler.GetAlphaCumulativeProduct(1), 1e-12);

        var sample = new Vector<double>(new[] { 0.7, -0.2 });
        var eps = new Vector<double>(new[] { 0.3, 0.5 });
        var result = scheduler.Step(eps, 1, sample, 0.0);
        // t = 1, prev = 1 − 2 < 0: ᾱ_prev = final_alpha_cumprod (1 or ᾱ₀); eta 0 gives √ᾱ_prev·x̂₀ + √(1 − ᾱ_prev)·ε.
        double prev = setAlphaToOne ? 1.0 : a0;
        for (int i = 0; i < 2; i++)
        {
            double x0 = (sample[i] - Math.Sqrt(1 - a1) * eps[i]) / Math.Sqrt(a1);
            Assert.Equal(Math.Sqrt(prev) * x0 + Math.Sqrt(1 - prev) * eps[i], result[i], 1e-12);
        }
    }
}
