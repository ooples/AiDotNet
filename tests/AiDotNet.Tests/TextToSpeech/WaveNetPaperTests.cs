using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.Vocoders;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// WaveNet follows its paper (van den Oord et al. 2016): μ-law classes and a softmax, dilated causal convolutions of width
/// 2 with dilations doubling to 512, gated units conditioned on the upsampled mel, teacher-forced likelihood training and
/// sample-by-sample generation.
/// </summary>
/// <remarks>
/// Before this change WaveNet's "synthesis" ran a generic layer stack once over the mel spectrogram: no μ-law
/// classes, no causality and no autoregressive sampling.
/// </remarks>
public class WaveNetPaperTests
{
    private static WaveNetOptions Options() => new()
    {
        MelChannels = 8, FftSize = 64, WindowSize = 64, HopSize = 16, SampleRate = 4000, MelMinFrequency = 0, MelMaxFrequency = 2000,
        UpsampleScales = [4, 4], NumDilatedLayers = 6, DilationCycle = 3, ResidualChannels = 8, GateChannels = 8, SkipChannels = 8,
        MuLawLevels = 32, SegmentSamples = 256,
    };

    private static WaveNet<double> CreateModel() => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 8, outputSize: 16) { RandomSeed = 8 },
        Options(),
        new AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>>(null));

    private static Tensor<double> Audio(int length = 512)
    {
        var audio = new Tensor<double>(new[] { length });
        for (int i = 0; i < length; i++) audio[i] = 0.4 * Math.Sin(2 * Math.PI * 250 * i / 4000.0);
        return audio;
    }

    private static Tensor<double> Frames(Tensor<double> mel, int count)
    {
        var slice = new Tensor<double>(new[] { 1, mel.Shape[1], count });
        for (int c = 0; c < mel.Shape[1]; c++)
            for (int f = 0; f < count; f++) slice[0, c, f] = mel[0, c, f];
        return slice;
    }

    [Fact(Timeout = 60000)]
    public async Task MuLaw_CompandsAndQuantizesTo256Levels()
    {
        await Task.Yield();
        Assert.Equal(128, MuLaw.Encode(0.0, 256));
        Assert.Equal(0, MuLaw.Encode(-1.0, 256));
        Assert.Equal(255, MuLaw.Encode(1.0, 256));
        // Companding spends more levels near zero: a quiet sample round-trips more precisely than a loud one.
        Assert.True(Math.Abs(MuLaw.Decode(MuLaw.Encode(0.01, 256), 256) - 0.01) < 5e-4);
        Assert.True(Math.Abs(MuLaw.Decode(MuLaw.Encode(0.9, 256), 256) - 0.9) < 0.02);
    }

    [Fact(Timeout = 60000)]
    public async Task ReceptiveField_IsThatOfWidthTwoConvolutionsWithDoublingDilations()
    {
        await Task.Yield();
        // Each 1, 2, …, 512 block of width-2 convolutions sees 1024 samples (§2.1); the input causal convolution adds one.
        var paper = new WaveNet<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 80, outputSize: 256),
            new WaveNetOptions { NumDilatedLayers = 10, ResidualChannels = 2, GateChannels = 2, SkipChannels = 2 });
        Assert.Equal(1025, paper.ReceptiveField);
        Assert.Equal(1 + (1 + 1 + 2 + 4 + 1 + 2 + 4), CreateModel().ReceptiveField);
    }

    [Fact(Timeout = 60000)]
    public async Task Predictions_AreCausal()
    {
        await Task.Yield();
        var model = CreateModel();
        var audio = Audio(256);
        var mel = Frames(model.ComputeMel(Audio()), 16);
        var before = model.LogProbabilities(mel, audio);
        var changed = new Tensor<double>(new[] { 256 });
        for (int i = 0; i < 256; i++) changed[i] = i < 100 ? audio[i] : -audio[i];
        var after = model.LogProbabilities(mel, changed);
        // Samples 0..100 are predicted from samples before 100 only.
        for (int t = 0; t <= 100; t++)
            for (int c = 0; c < 32; c++) Assert.Equal(before[0, c, t], after[0, c, t], 12);
        Assert.Contains(Enumerable.Range(101, 155), t => Enumerable.Range(0, 32).Any(c => Math.Abs(before[0, c, t] - after[0, c, t]) > 1e-9));
    }

    [Fact(Timeout = 60000)]
    public async Task IncrementalSteps_ReproduceTheTeacherForcedNetwork()
    {
        await Task.Yield();
        var engine = AiDotNetEngine.Current;
        var network = new WaveNetNetwork<double>(engine, new Random(3), 16, 4, 6, 6, 6, 5, 3, 2, new[] { 2, 2 });
        var random = new Random(4);
        int samples = 40;
        var previous = new Tensor<double>(new[] { 1, 16, samples });
        var classes = Enumerable.Range(0, samples).Select(_ => random.Next(16)).ToArray();
        for (int t = 0; t < samples; t++) previous[0, classes[t], t] = 1;
        var condition = new Tensor<double>(new[] { 1, 4, samples });
        for (int i = 0; i < condition.Length; i++) condition[i] = random.NextDouble() - 0.5;
        var full = network.Forward(previous, condition);
        var state = network.NewState();
        for (int t = 0; t < samples; t++)
        {
            var oneHot = new Tensor<double>(new[] { 1, 16, 1 });
            oneHot[0, classes[t], 0] = 1;
            var column = new Tensor<double>(new[] { 1, 4, 1 });
            for (int c = 0; c < 4; c++) column[0, c, 0] = condition[0, c, t];
            var step = network.Step(state, oneHot, column);
            for (int c = 0; c < 16; c++) Assert.Equal(full[0, c, t], step[0, c, 0], 9);
        }
    }

    [Fact(Timeout = 300000)]
    public async Task Training_ReducesTheCrossEntropy()
    {
        await Task.Yield();
        var model = CreateModel();
        var provider = (AiDotNet.Interfaces.ITrainingObjectiveProvider<double>)model;
        var audio = Audio(256);
        var mel = Frames(model.ComputeMel(Audio()), 16);
        double before = provider.EvaluateTrainingObjective(mel, audio);
        for (int i = 0; i < 20; i++) model.Train(mel, audio);
        double after = provider.EvaluateTrainingObjective(mel, audio);
        Assert.True(after < before, $"The cross-entropy did not fall ({before} -> {after}).");
    }

    [Fact(Timeout = 120000)]
    public async Task Synthesis_DrawsOneMuLawClassPerSample_Repeatably()
    {
        await Task.Yield();
        var model = CreateModel();
        var mel = Frames(model.ComputeMel(Audio()), 4);
        var wave = model.MelToWaveform(mel);
        Assert.Equal(64, wave.Length);
        var levels = Enumerable.Range(0, 32).Select(q => MuLaw.Decode(q, 32)).ToArray();
        for (int i = 0; i < wave.Length; i++) Assert.Contains(levels, v => Math.Abs(v - wave[i]) < 1e-12);
        Assert.Equal(wave.ToVector().ToArray(), model.MelToWaveform(mel).ToVector().ToArray());
    }
}
