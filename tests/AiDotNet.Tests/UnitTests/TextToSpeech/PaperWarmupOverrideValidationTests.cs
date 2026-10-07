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

    private static NeuralNetworkArchitecture<double> CreateArchitecture() =>
        new(inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.TextGeneration,
            inputSize: 4, outputSize: 16);
}
