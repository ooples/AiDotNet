using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.Vocoders;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// Vocos generates and trains as its paper specifies (Siuzdak 2024): a frame-rate ConvNeXt backbone, a magnitude and
/// unit-circle phase head inverted by the centred iSTFT, and hinge-loss adversarial training with feature matching and
/// the mel L1.
/// </summary>
/// <remarks>
/// Before this change Vocos used a "same"-padded inverse STFT (frames × hop samples from the centred features of
/// (frames − 1) × hop) and trained by regression without discriminators.
/// </remarks>
public class VocosPaperTests
{
    private static VocosOptions Options() => new()
    {
        MelChannels = 8, FftSize = 64, HopSize = 16, SampleRate = 4000, ConvNeXtDim = 16, NumBackboneBlocks = 2, IntermediateDim = 32,
        DiscriminatorPeriods = [2, 3], DiscriminatorWidthDivisor = 32, ResolutionDiscriminatorChannels = 4,
        ResolutionFftSizes = [64, 128], ResolutionHopSizes = [8, 16], ResolutionWindowSizes = [40, 80], SegmentSize = 512,
    };

    private static Vocos<double> CreateModel() => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 8, outputSize: 16) { RandomSeed = 6 },
        Options(),
        new AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>>(null));

    private static Tensor<double> Audio()
    {
        var audio = new Tensor<double>(new[] { 512 });
        for (int i = 0; i < 512; i++) audio[i] = 0.4 * Math.Sin(2 * Math.PI * 330 * i / 4000.0);
        return audio;
    }

    [Fact(Timeout = 60000)]
    public async Task CentredFeatures_RoundTripToTheRecordingsLength()
    {
        await Task.Yield();
        var model = CreateModel();
        var mel = model.ComputeMel(Audio());
        Assert.Equal(512 / 16 + 1, mel.Shape[2]);
        Assert.Equal(512, model.MelToWaveform(mel).Length);
    }

    [Fact(Timeout = 300000)]
    public async Task AdversarialTraining_ReducesTheMelReconstructionError()
    {
        await Task.Yield();
        var model = CreateModel();
        var provider = (AiDotNet.Interfaces.ITrainingObjectiveProvider<double>)model;
        var audio = Audio();
        var mel = model.ComputeMel(audio);
        double before = provider.EvaluateTrainingObjective(mel, audio);
        for (int i = 0; i < 15; i++) model.Train(mel, audio);
        double after = provider.EvaluateTrainingObjective(mel, audio);
        Assert.True(after < before, $"The mel loss did not fall ({before} -> {after}).");
    }
}
