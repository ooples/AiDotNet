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
/// WaveGlow follows its paper (Prenger et al. 2019): an invertible flow of 1×1 convolutions and affine couplings over
/// squeezed audio with early outputs, trained by the likelihood −z²/2σ² + Σ log s + Σ log|det W|, sampled by inverting the
/// flow from z ~ N(0, σ²).
/// </summary>
/// <remarks>
/// Before this change WaveGlow had no flow: its synthesis ran a generic layer stack forward and training regressed it,
/// with no invertibility and no likelihood.
/// </remarks>
public class WaveGlowPaperTests
{
    private static WaveGlowOptions Options() => new()
    {
        MelChannels = 8, FftSize = 64, WindowSize = 64, HopSize = 16, SampleRate = 4000, MelMaxFrequency = 2000,
        UpsampleKernel = 64, GroupSize = 8, NumFlows = 6, EarlyOutputEvery = 2, EarlyOutputChannels = 2,
        NumWaveNetLayers = 2, ResidualChannels = 8, GateChannels = 8, SkipChannels = 4, SegmentSamples = 256,
    };

    private static WaveGlow<double> CreateModel() => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 8, outputSize: 16) { RandomSeed = 8 },
        Options(),
        new AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>>(null));

    private static Tensor<double> Audio()
    {
        var audio = new Tensor<double>(new[] { 512 });
        for (int i = 0; i < 512; i++) audio[i] = 0.4 * Math.Sin(2 * Math.PI * 250 * i / 4000.0) + 0.05 * Math.Sin(i * 1.3);
        return audio;
    }

    private static Tensor<double> Frames(Tensor<double> mel, int count)
    {
        var slice = new Tensor<double>(new[] { 1, mel.Shape[1], count });
        for (int c = 0; c < mel.Shape[1]; c++)
            for (int f = 0; f < count; f++) slice[0, c, f] = mel[0, c, f];
        return slice;
    }

    private static Tensor<double> Samples(Tensor<double> audio, int count)
    {
        var s = new Tensor<double>(new[] { count });
        for (int i = 0; i < count; i++) s[i] = audio[i];
        return s;
    }

    [Fact(Timeout = 120000)]
    public async Task Flow_IsExactlyInvertible_AfterTraining()
    {
        await Task.Yield();
        var model = CreateModel();
        var audio = Audio();
        var mel = Frames(model.ComputeMel(audio), 16);
        var clip = Samples(audio, 256);
        // Train a few steps so the couplings are no longer the identity they start as.
        for (int i = 0; i < 5; i++) model.Train(mel, clip);
        var (z, _) = model.Encode(mel, clip);
        Assert.Equal(new[] { 1, 8, 32 }, z.Shape.ToArray());
        var back = model.Decode(mel, z);
        for (int i = 0; i < 256; i++) Assert.Equal(clip[i], back[0, 0, i], 6);
    }

    [Fact(Timeout = 60000)]
    public async Task UntrainedFlow_IsAVolumePreservingRotation_SoTheLossIsTheGaussianEnergy()
    {
        await Task.Yield();
        var model = CreateModel();
        var provider = (AiDotNet.Interfaces.ITrainingObjectiveProvider<double>)model;
        var audio = Audio();
        var mel = Frames(model.ComputeMel(audio), 16);
        var clip = Samples(audio, 256);
        // The couplings' end projections start at zero (log s = 0, shift 0) and the 1×1 convolutions are rotations
        // (log|det W| = 0, ‖Wx‖ = ‖x‖): the loss is Σx² / (2σ² · samples) = mean(x²) at σ² = 0.5.
        double meanSquare = Enumerable.Range(0, 256).Average(i => clip[i] * clip[i]);
        Assert.Equal(meanSquare, provider.EvaluateTrainingObjective(mel, clip), 9);
    }

    [Fact(Timeout = 300000)]
    public async Task Training_IncreasesTheLikelihood()
    {
        await Task.Yield();
        var model = CreateModel();
        var provider = (AiDotNet.Interfaces.ITrainingObjectiveProvider<double>)model;
        var audio = Audio();
        var mel = Frames(model.ComputeMel(audio), 16);
        var clip = Samples(audio, 256);
        double before = provider.EvaluateTrainingObjective(mel, clip);
        for (int i = 0; i < 20; i++) model.Train(mel, clip);
        double after = provider.EvaluateTrainingObjective(mel, clip);
        Assert.True(after < before, $"The negative log-likelihood did not fall ({before} -> {after}).");
    }

    [Fact(Timeout = 120000)]
    public async Task Synthesis_DrawsZAtTheInferenceSigma_AndIsRepeatable()
    {
        await Task.Yield();
        var model = CreateModel();
        var mel = Frames(model.ComputeMel(Audio()), 16);
        var wave = model.MelToWaveform(mel);
        Assert.Equal(256, wave.Length);
        // The untrained flow is a rotation, so the samples are N(0, 0.6²) noise.
        double std = Math.Sqrt(Enumerable.Range(0, 256).Average(i => wave[i] * wave[i]));
        Assert.InRange(std, 0.5, 0.7);
        Assert.Equal(wave.ToVector().ToArray(), model.MelToWaveform(mel).ToVector().ToArray());
    }
}
