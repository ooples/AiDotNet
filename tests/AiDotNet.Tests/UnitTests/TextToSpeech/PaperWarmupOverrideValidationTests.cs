using System;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.FlowDiffusion;
using Xunit;

namespace AiDotNet.Tests.UnitTests.TextToSpeech;

/// <summary>
/// PaperOptimizerFactory ignores a non-positive warmup override, so a model that forwarded one would train on
/// the paper's warmup (32000 steps for MaskGCT, 5000 for NaturalSpeech 3) instead of the value its caller set.
/// </summary>
public sealed class PaperWarmupOverrideValidationTests
{
    // The exception is checked down to its parameter and rejected value: construction can throw
    // ArgumentOutOfRangeException for other reasons, and only these two prove it came from the warmup check.
    [Theory]
    [InlineData(0)]
    [InlineData(-5)]
    [InlineData(int.MinValue)]
    public void MaskGCT_RejectsANonPositiveWarmup(int warmupSteps)
    {
        var error = Assert.Throws<ArgumentOutOfRangeException>(
            () => new MaskGCT<double>(CreateArchitecture(), SmallMaskGct(warmupSteps)));
        Assert.Equal("options", error.ParamName);
        Assert.Equal(warmupSteps, error.ActualValue);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-5)]
    [InlineData(int.MinValue)]
    public void NaturalSpeech3_RejectsANonPositiveWarmup(int warmupSteps)
    {
        var error = Assert.Throws<ArgumentOutOfRangeException>(
            () => new NaturalSpeech3<double>(CreateSpectrogramArchitecture(), SmallNaturalSpeech3(warmupSteps)));
        Assert.Equal("options", error.ParamName);
        Assert.Equal(warmupSteps, error.ActualValue);
    }

    // The boundary: the check is `<= 0`, so 1 must be accepted. A regression to `< 1` or `<= 1` would
    // still reject 0 and -5, which is why the smallest positive value is tested on its own.
    [Fact]
    public void MaskGCT_AcceptsTheSmallestPositiveWarmup()
    {
        var model = new MaskGCT<double>(CreateArchitecture(), SmallMaskGct(warmupSteps: 1));
        Assert.NotNull(model.TrainingOptimizer);
    }

    [Fact]
    public void NaturalSpeech3_AcceptsTheSmallestPositiveWarmup()
    {
        var model = new NaturalSpeech3<double>(CreateSpectrogramArchitecture(), SmallNaturalSpeech3(warmupSteps: 1));
        Assert.NotNull(model.TrainingOptimizer);
    }

    [Fact]
    public void MaskGCT_ForwardsAPositiveWarmupToItsOptimizer()
    {
        AssertWarmupReachesTheSchedule(
            new MaskGCT<double>(CreateArchitecture(), SmallMaskGct(warmupSteps: 2)).TrainingOptimizer,
            new MaskGCT<double>(CreateArchitecture(), SmallMaskGct(warmupSteps: null)).TrainingOptimizer);
    }

    [Fact]
    public void NaturalSpeech3_ForwardsAPositiveWarmupToItsOptimizer()
    {
        AssertWarmupReachesTheSchedule(
            new NaturalSpeech3<double>(CreateSpectrogramArchitecture(), SmallNaturalSpeech3(warmupSteps: 2)).TrainingOptimizer,
            new NaturalSpeech3<double>(CreateSpectrogramArchitecture(), SmallNaturalSpeech3(warmupSteps: null)).TrainingOptimizer);
    }

    /// <summary>
    /// A 2-step warmup is below the recipe's peak at step 1 and exactly at it by step 2 -- which pins the
    /// forwarded value itself, not merely "shorter than the default". The default is not a fixed 32000 or
    /// 5000 steps here: the recipe factory rescales a paper warmup to its share of the configured run, so
    /// at these fixture sizes it reaches 2.8e-5 of a 1e-4 peak by step 2. The comparison against it only
    /// shows the override is not the default; the step-1/step-2 shape is what shows it is 2.
    /// </summary>
    private static void AssertWarmupReachesTheSchedule(
        IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>? shortWarmup,
        IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>? paperWarmup)
    {
        // Read off the schedule's own shape, not BaseLearningRate, and without pinning the exact index
        // the ramp tops out at: LinearWarmupScheduler counts from step 0 and peaks at index 2, while
        // NoamHoldAnnealingScheduler counts from 1 (trainingStep = step + 1) and peaks at index 1. In
        // both, a 2-step warmup is still rising at step 0 and has reached its maximum by step 2.
        var shortSchedule = ScheduleOf(shortWarmup);
        double atZero = shortSchedule.GetLearningRateAtStep(0);
        double atTwo = shortSchedule.GetLearningRateAtStep(2);
        double highest = 0;
        for (int step = 0; step <= 10; step++)
            highest = Math.Max(highest, shortSchedule.GetLearningRateAtStep(step));
        double paperAtTwo = ScheduleOf(paperWarmup).GetLearningRateAtStep(2);

        Assert.True(atTwo > 0, $"the short warmup reached no learning rate at step 2 ({atTwo})");
        Assert.True(atZero < atTwo, $"step 0: {atZero} is not below step 2's {atTwo}; there is no warmup at all");
        Assert.True(atTwo >= highest * (1 - 1e-12),
            $"step 2: {atTwo} is below the schedule's maximum {highest}; the warmup is longer than 2 steps");
        Assert.True(paperAtTwo < atTwo,
            $"step 2: {atTwo} with a 2-step warmup against {paperAtTwo} with the default; the override did not apply");
    }

    private static ILearningRateScheduler ScheduleOf(
        IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>? trainingOptimizer)
    {
        var optimizer = Assert.IsAssignableFrom<GradientBasedOptimizerBase<double, Tensor<double>, Tensor<double>>>(
            trainingOptimizer);
        var schedule = optimizer.LearningRateScheduler;
        if (schedule is null)
            throw new Xunit.Sdk.XunitException("The training optimizer has no learning-rate schedule.");
        return schedule;
    }

    // Fixture-sized models, as the generated tests build them: the warmup is a property of the optimizer
    // recipe, not of width, and paper-sized layers would make this a heavy test for no added coverage.
    // A null warmup keeps the options' own default, which is the paper's.
    private static MaskGCTOptions SmallMaskGct(int? warmupSteps)
    {
        var options = new MaskGCTOptions
        {
            NumCodebooks = 2, CodebookSize = 16, TextEncoderDim = 32, LLMDim = 64, NumEncoderLayers = 1,
            NumLLMLayers = 2, NumHeads = 4, MaxTextLength = 8, MaxCodecFrames = 8, DropoutRate = 0.0,
        };
        if (warmupSteps is int steps) options.WarmupSteps = steps;
        return options;
    }

    private static NaturalSpeech3Options SmallNaturalSpeech3(int? warmupSteps)
    {
        var options = new NaturalSpeech3Options
        {
            HiddenDim = 32, EncoderDim = 32, DecoderDim = 32, DiffusionDim = 32, MelChannels = 16,
            NumEncoderLayers = 1, NumDiffusionSteps = 2, NumHeads = 4, DropoutRate = 0.0, MaxTextLength = 16,
        };
        if (warmupSteps is int steps) options.WarmupSteps = steps;
        return options;
    }

    private static NeuralNetworkArchitecture<double> CreateSpectrogramArchitecture() =>
        new(inputType: InputType.TwoDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 64, inputWidth: 16, inputDepth: 1, outputSize: 16);

    private static NeuralNetworkArchitecture<double> CreateArchitecture() =>
        new(inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.TextGeneration,
            inputSize: 4, outputSize: 16);
}
