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
/// UnivNet generates and trains as its paper specifies (Jang et al. 2021): noise shaped by location-variable convolutions
/// with frame-wise predicted kernels, the weighted multi-resolution STFT loss alone for the first steps, then LSGAN
/// against multi-resolution spectrogram and multi-period discriminators.
/// </summary>
/// <remarks>
/// Before this change UnivNet built a HiFi-GAN generator and trained it by regression; it had no kernel predictor,
/// location-variable convolution, noise input or discriminators.
/// </remarks>
public class UnivNetPaperTests
{
    private static UnivNetOptions Options(long pretraining = 200_000) => new()
    {
        MelChannels = 8, FftSize = 64, WindowSize = 64, HopSize = 16, SampleRate = 4000, MelMaxFrequency = 2000,
        UpsampleRates = [4, 4], NoiseDim = 4, ChannelSize = 4, Dilations = [1, 3], KernelPredictorHidden = 8,
        DiscriminatorPeriods = [2, 3], PeriodDiscriminatorChannels = [8, 8, 8, 8, 8], ResolutionDiscriminatorChannels = 4,
        StftFftSizes = [64, 128], StftHopSizes = [8, 16], StftWindowSizes = [40, 80], SegmentSize = 512, InferencePaddingFrames = 2,
        PretrainingSteps = pretraining,
    };

    private static UnivNet<double> CreateModel(UnivNetOptions options) => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 8, outputSize: 16) { RandomSeed = 4 },
        options,
        new AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>>(null));

    private static Tensor<double> Audio()
    {
        var audio = new Tensor<double>(new[] { 512 });
        for (int i = 0; i < 512; i++) audio[i] = 0.4 * Math.Sin(2 * Math.PI * 300 * i / 4000.0) + 0.2 * Math.Sin(2 * Math.PI * 1100 * i / 4000.0);
        return audio;
    }

    [Fact(Timeout = 60000)]
    public async Task Synthesis_GivesHopSamplesPerFrame_TrimmingThePaddingFrames_Repeatably()
    {
        await Task.Yield();
        var model = CreateModel(Options());
        var mel = new Tensor<double>(new[] { 1, 8, 5 });
        for (int i = 0; i < mel.Length; i++) mel[i] = Math.Cos(0.4 * i) - 3;
        var first = model.MelToWaveform(mel);
        Assert.Equal(5 * 16, first.Length);
        Assert.Equal(first.ToVector().ToArray(), model.MelToWaveform(mel).ToVector().ToArray());
        for (int i = 0; i < first.Length; i++) Assert.InRange(first[i], -1.0, 1.0);
    }

    [Fact(Timeout = 300000)]
    public async Task Pretraining_ReducesTheWeightedStftLoss_LeavingTheDiscriminatorsUntouched()
    {
        await Task.Yield();
        var model = CreateModel(Options());
        var provider = (AiDotNet.Interfaces.ITrainingObjectiveProvider<double>)model;
        var audio = Audio();
        var mel = model.ComputeMel(audio);
        double before = provider.EvaluateTrainingObjective(mel, audio);
        var start = model.GetParameters().ToArray();
        for (int i = 0; i < 20; i++) model.Train(mel, audio);
        double after = provider.EvaluateTrainingObjective(mel, audio);
        Assert.True(after < before, $"The STFT loss did not fall ({before} -> {after}).");
        var end = model.GetParameters().ToArray();
        for (int i = start.Length - 20; i < start.Length; i++) Assert.Equal(start[i], end[i]);
    }

    [Fact(Timeout = 300000)]
    public async Task OnceAdversarial_BothNetworksTrain()
    {
        await Task.Yield();
        var model = CreateModel(Options(pretraining: 0));
        var audio = Audio();
        var start = model.GetParameters().ToArray();
        model.Train(model.ComputeMel(audio), audio);
        var end = model.GetParameters().ToArray();
        Assert.Contains(Enumerable.Range(start.Length - 20, 20), i => start[i] != end[i]);
        Assert.Contains(Enumerable.Range(0, 20), i => start[i] != end[i]);
    }
}
