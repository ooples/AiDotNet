using System;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
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
    [Theory]
    [InlineData(0)]
    [InlineData(-5)]
    public void MaskGCT_RejectsANonPositiveWarmup(int warmupSteps)
    {
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new MaskGCT<double>(CreateArchitecture(), new MaskGCTOptions { WarmupSteps = warmupSteps }));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-5)]
    public void NaturalSpeech3_RejectsANonPositiveWarmup(int warmupSteps)
    {
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new NaturalSpeech3<double>(CreateArchitecture(), new NaturalSpeech3Options { WarmupSteps = warmupSteps }));
    }

    [Fact]
    public void MaskGCT_ForwardsAPositiveWarmupToItsOptimizer()
    {
        AssertWarmupReachesTheSchedule(
            new MaskGCT<double>(CreateArchitecture(), SmallMaskGct(warmupSteps: 2)),
            new MaskGCT<double>(CreateArchitecture(), SmallMaskGct(warmupSteps: null)));
    }

    [Fact]
    public void NaturalSpeech3_ForwardsAPositiveWarmupToItsOptimizer()
    {
        AssertWarmupReachesTheSchedule(
            new NaturalSpeech3<double>(CreateSpectrogramArchitecture(), SmallNaturalSpeech3(warmupSteps: 2)),
            new NaturalSpeech3<double>(CreateSpectrogramArchitecture(), SmallNaturalSpeech3(warmupSteps: null)));
    }

    /// <summary>
    /// Two steps into a 2-step warmup the rate has reached the recipe's peak; two steps into the paper's
    /// warmup (32000 or 5000 steps) it is still a sliver of it. The ratio is what proves the override
    /// reached the optimizer the model trains with, whatever the schedule's shape after the ramp.
    /// </summary>
    private static void AssertWarmupReachesTheSchedule(object shortWarmup, object paperWarmup)
    {
        double shortRate = ScheduleOf(shortWarmup).GetLearningRateAtStep(2);
        double paperRate = ScheduleOf(paperWarmup).GetLearningRateAtStep(2);

        Assert.True(shortRate > 0, $"the short warmup reached no learning rate at step 2 ({shortRate})");
        Assert.True(shortRate > 100 * paperRate,
            $"step 2: {shortRate} with a 2-step warmup against {paperRate} with the paper's; the override did not apply");
    }

    private static AiDotNet.LearningRateSchedulers.ILearningRateScheduler ScheduleOf(object model)
    {
        var field = model.GetType().GetField(
            "_optimizer", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
        if (field is null)
            throw new Xunit.Sdk.XunitException($"{model.GetType().Name} has no _optimizer field to inspect.");
        var optimizer = Assert.IsAssignableFrom<AiDotNet.Optimizers.GradientBasedOptimizerBase<
            double, AiDotNet.Tensors.LinearAlgebra.Tensor<double>, AiDotNet.Tensors.LinearAlgebra.Tensor<double>>>(
            field.GetValue(model));
        var schedule = optimizer.LearningRateScheduler;
        if (schedule is null)
            throw new Xunit.Sdk.XunitException($"{model.GetType().Name}'s optimizer has no learning-rate schedule.");
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
