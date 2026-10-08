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
/// Parallel WaveGAN trains and generates as its paper specifies (Yamamoto et al. 2020): a non-causal WaveNet over Gaussian
/// noise conditioned on the upsampled mel spectrogram, the multi-resolution STFT loss alone while the discriminator is
/// fixed, then LSGAN adversarial training with λ_adv = 4.
/// </summary>
/// <remarks>
/// Before this change Parallel WaveGAN ran a time-preserving WaveNet stack on the mel frames (one output sample per
/// frame) and trained it by regression; it had no noise input, no upsampling, no STFT loss and no discriminator.
/// </remarks>
public class ParallelWaveGANPaperTests
{
    private static ParallelWaveGANOptions Options(long discriminatorStart = 100_000) => new()
    {
        MelChannels = 8, FftSize = 64, WindowSize = 48, HopSize = 16, SampleRate = 4000, UpsampleRates = [4, 4],
        MelMinFrequency = 0, MelMaxFrequency = 2000, NumLayers = 4, NumStacks = 2, ResidualChannels = 8, GateChannels = 8,
        SkipChannels = 8, DiscriminatorLayers = 4, DiscriminatorChannels = 8, StftFftSizes = [64, 128], StftWindowSizes = [40, 80],
        StftHopSizes = [8, 16], SegmentSize = 512, DiscriminatorStartStep = discriminatorStart,
    };

    private static ParallelWaveGAN<double> CreateModel(ParallelWaveGANOptions options) => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 8, outputSize: 16) { RandomSeed = 3 },
        options,
        new AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>>(null));

    private static Tensor<double> Audio()
    {
        var audio = new Tensor<double>(new[] { 512 });
        for (int i = 0; i < 512; i++) audio[i] = 0.4 * Math.Sin(2 * Math.PI * 220 * i / 4000.0) + 0.2 * Math.Sin(2 * Math.PI * 900 * i / 4000.0);
        return audio;
    }

    [Fact(Timeout = 60000)]
    public async Task Generator_ShapesNoiseIntoHopSamplesPerFrame_Repeatably()
    {
        await Task.Yield();
        var model = CreateModel(Options());
        var mel = new Tensor<double>(new[] { 1, 8, 6 });
        for (int i = 0; i < mel.Length; i++) mel[i] = Math.Sin(0.3 * i);
        var first = model.MelToWaveform(mel);
        Assert.Equal(6 * 16, first.Length);
        Assert.Equal(first.ToVector().ToArray(), model.MelToWaveform(mel).ToVector().ToArray());
    }

    [Fact(Timeout = 300000)]
    public async Task WhileTheDiscriminatorIsFixed_TheStftLossFalls_AndOnlyTheGeneratorTrains()
    {
        await Task.Yield();
        var model = CreateModel(Options());
        var provider = (AiDotNet.Interfaces.ITrainingObjectiveProvider<double>)model;
        var audio = Audio();
        var full = model.ComputeMel(audio);
        var mel = new Tensor<double>(new[] { 1, 8, 32 });
        for (int c = 0; c < 8; c++)
            for (int f = 0; f < 32; f++) mel[0, c, f] = full[0, c, f];
        double before = provider.EvaluateTrainingObjective(mel, audio);
        var discriminatorStart = GanVocoderWeights.Of(model.DiscriminatorLayers);
        for (int i = 0; i < 20; i++) model.Train(mel, audio);
        double after = provider.EvaluateTrainingObjective(mel, audio);
        Assert.True(after < before, $"The STFT loss did not fall ({before} -> {after}).");
        // Every discriminator weight is untouched during pre-training.
        Assert.Equal(discriminatorStart, GanVocoderWeights.Of(model.DiscriminatorLayers));
    }

    [Fact(Timeout = 120000)]
    public async Task OnceAdversarial_TheDiscriminatorTrainsAfterTheGenerator()
    {
        await Task.Yield();
        var model = CreateModel(Options(discriminatorStart: 0));
        var audio = Audio();
        var generatorStart = GanVocoderWeights.Of(model.GeneratorLayers);
        var discriminatorStart = GanVocoderWeights.Of(model.DiscriminatorLayers);
        model.Train(model.ComputeMel(audio), audio);
        Assert.True(GanVocoderWeights.AnyChanged(discriminatorStart, GanVocoderWeights.Of(model.DiscriminatorLayers)),
            "An adversarial step left every discriminator weight where it was.");
        Assert.True(GanVocoderWeights.AnyChanged(generatorStart, GanVocoderWeights.Of(model.GeneratorLayers)),
            "An adversarial step left every generator weight where it was.");
    }
}
