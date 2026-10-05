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
/// Multi-band MelGAN trains and generates as its paper specifies (Yang et al. 2021): four PQMF sub-bands from a shared
/// generator, full- and sub-band multi-resolution STFT losses, generator-only pre-training, then LSGAN adversarial
/// training.
/// </summary>
/// <remarks>
/// Before this change Multi-band MelGAN built a full-band HiFi-GAN generator and refused its own band count; it had no
/// filter bank, no STFT loss and no discriminators.
/// </remarks>
public class MultiBandMelGANPaperTests
{
    private static MultiBandMelGANOptions Options(long pretraining = 200_000) => new()
    {
        MelChannels = 8, FftSize = 64, WindowSize = 48, HopSize = 16, SampleRate = 4000, UpsampleRates = [2, 2],
        UpsampleInitialChannels = 16, ResidualLayers = 2, DiscriminatorWidthDivisor = 16, MelMinFrequency = 0, MelMaxFrequency = 2000,
        FullBandFftSizes = [64, 128], FullBandWindowSizes = [40, 80], FullBandHopSizes = [8, 16],
        SubBandFftSizes = [32, 16], SubBandWindowSizes = [16, 8], SubBandHopSizes = [4, 2], SegmentSize = 512,
        PretrainingSteps = pretraining,
    };

    private static MultiBandMelGAN<double> CreateModel(MultiBandMelGANOptions options) => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 8, outputSize: 16) { RandomSeed = 5 },
        options,
        new AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>>(null));

    private static Tensor<double> Audio()
    {
        var audio = new Tensor<double>(new[] { 512 });
        for (int i = 0; i < 512; i++) audio[i] = 0.4 * Math.Sin(2 * Math.PI * 220 * i / 4000.0) + 0.2 * Math.Sin(2 * Math.PI * 1230 * i / 4000.0);
        return audio;
    }

    [Fact(Timeout = 60000)]
    public async Task Pqmf_SplitsIntoFourQuarterRateBands_AndReconstructsNearlyPerfectly()
    {
        await Task.Yield();
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var pqmf = new PseudoQmf<double>(engine);
        var x = new Tensor<double>(new[] { 1, 1, 1024 });
        for (int i = 0; i < 1024; i++) x[0, 0, i] = Math.Sin(0.05 * i) + 0.5 * Math.Sin(1.9 * i);
        var bands = pqmf.Analysis(x);
        Assert.Equal(new[] { 1, 4, 256 }, bands.Shape.ToArray());
        var y = pqmf.Synthesis(bands);
        Assert.Equal(1024, y.Length);
        // Away from the edges the 63-coefficient bank reconstructs the input to about −40 dB.
        double error = 0, energy = 0;
        for (int i = 100; i < 924; i++)
        {
            double d = y[0, 0, i] - x[0, 0, i];
            error += d * d;
            energy += x[0, 0, i] * x[0, 0, i];
        }
        Assert.True(error / energy < 1e-3, $"Relative reconstruction error {error / energy}.");
    }

    [Fact(Timeout = 60000)]
    public async Task Generator_PredictsBands_WhoseSynthesisIsBandsTimesUpsamplingSamplesPerFrame()
    {
        await Task.Yield();
        var model = CreateModel(Options());
        var mel = new Tensor<double>(new[] { 1, 8, 5 });
        Assert.Equal(5 * 16, model.MelToWaveform(mel).Length);
    }

    [Fact(Timeout = 300000)]
    public async Task Pretraining_ReducesTheMultiResolutionStftLoss_WithoutTouchingTheDiscriminators()
    {
        await Task.Yield();
        var model = CreateModel(Options());
        var provider = (AiDotNet.Interfaces.ITrainingObjectiveProvider<double>)model;
        var audio = Audio();
        var mel = model.ComputeMel(audio);
        var frames = new Tensor<double>(new[] { 1, 8, 32 });
        for (int c = 0; c < 8; c++)
            for (int f = 0; f < 32; f++) frames[0, c, f] = mel[0, c, f];
        double before = provider.EvaluateTrainingObjective(frames, audio);
        var start = model.GetParameters().ToArray();
        for (int i = 0; i < 20; i++) model.Train(frames, audio);
        double after = provider.EvaluateTrainingObjective(frames, audio);
        Assert.True(after < before, $"The STFT loss did not fall ({before} -> {after}).");
        // The generator's parameters come first; the discriminators' (the tail) are untouched during pre-training.
        var end = model.GetParameters().ToArray();
        int discriminatorTail = 50;
        for (int i = start.Length - discriminatorTail; i < start.Length; i++) Assert.Equal(start[i], end[i]);
    }

    [Fact(Timeout = 300000)]
    public async Task AfterPretraining_TheDiscriminatorsTrain()
    {
        await Task.Yield();
        var model = CreateModel(Options(pretraining: 0));
        var audio = Audio();
        var start = model.GetParameters().ToArray();
        model.Train(model.ComputeMel(audio), audio);
        var end = model.GetParameters().ToArray();
        Assert.Contains(Enumerable.Range(start.Length - 50, 50), i => start[i] != end[i]);
    }
}
