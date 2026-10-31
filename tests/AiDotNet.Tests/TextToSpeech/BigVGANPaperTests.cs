using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.Vocoders;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// BigVGAN generates and trains as its paper specifies (Lee et al. 2023): Snake activations made anti-aliased by 2× Kaiser
/// sinc resampling, HiFi-GAN's generator otherwise, period and resolution discriminators, HiFi-GAN's losses.
/// </summary>
/// <remarks>
/// Before this change BigVGAN was a dense [T, mel] → [T, 1] regressor with no upsampling, activations or discriminators of
/// the paper.
/// </remarks>
public class BigVGANPaperTests
{
    private static BigVGANOptions Options() => new()
    {
        MelChannels = 8, FftSize = 64, WindowSize = 64, HopSize = 16, SampleRate = 4000, MelMaxFrequency = 2000,
        UpsampleRates = [4, 4], UpsampleKernelSizes = [8, 8], UpsampleInitialChannels = 16, ResblockKernelSizes = [3],
        ResblockDilationSizes = [[1, 3]], DiscriminatorPeriods = [2, 3], DiscriminatorWidthDivisor = 32,
        ResolutionDiscriminatorChannels = 4, ResolutionFftSizes = [64, 128], ResolutionHopSizes = [8, 16], ResolutionWindowSizes = [40, 80],
        SegmentSize = 512,
    };

    private static BigVGAN<double> CreateModel() => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 8, outputSize: 16) { RandomSeed = 2 },
        Options(),
        new AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>>(null));

    [Fact(Timeout = 60000)]
    public async Task Snake_IsIdentityPlusSineSquaredOverAlpha()
    {
        await Task.Yield();
        var snake = new SnakeLayer<double>(2);
        var x = new Tensor<double>(new[] { 1, 2, 3 });
        double[] values = { -1.2, 0.0, 0.7, 2.0, -0.3, 1.1 };
        for (int i = 0; i < 6; i++) x[i] = values[i];
        var y = snake.Forward(x);
        // The reference divides by α + 1e-9 (no_div_by_zero).
        for (int i = 0; i < 6; i++) Assert.Equal(values[i] + Math.Pow(Math.Sin(values[i]), 2) / (1 + 1e-9), y[i], 12);
    }

    [Fact(Timeout = 60000)]
    public async Task AntiAliasingFilter_IsAUnitGainKaiserSinc_AndKeepsTheLength()
    {
        await Task.Yield();
        var filter = AntiAliasedSnake<double>.KaiserSinc(0.25, 0.3, 12);
        Assert.Equal(1.0, filter.Sum(), 12);
        for (int i = 0; i < 6; i++) Assert.Equal(filter[i], filter[11 - i], 12);   // symmetric
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var owned = new System.Collections.Generic.List<LayerBase<double>>();
        var act = new AntiAliasedSnake<double>(engine, owned, 3);
        var x = new Tensor<double>(new[] { 1, 3, 20 });
        for (int i = 0; i < x.Length; i++) x[i] = Math.Sin(0.3 * i);
        Assert.Equal(new[] { 1, 3, 20 }, act.Forward(x).Shape.ToArray());
    }

    [Fact(Timeout = 300000)]
    public async Task AdversarialTraining_ReducesTheMelReconstructionError()
    {
        await Task.Yield();
        var model = CreateModel();
        var provider = (AiDotNet.Interfaces.ITrainingObjectiveProvider<double>)model;
        var audio = new Tensor<double>(new[] { 512 });
        for (int i = 0; i < 512; i++) audio[i] = 0.4 * Math.Sin(2 * Math.PI * 220 * i / 4000.0);
        var mel = model.ComputeMel(audio);
        Assert.Equal(mel.Shape[2] * 16, model.MelToWaveform(mel).Length);
        double before = provider.EvaluateTrainingObjective(mel, audio);
        for (int i = 0; i < 15; i++) model.Train(mel, audio);
        double after = provider.EvaluateTrainingObjective(mel, audio);
        Assert.True(after < before, $"The mel loss did not fall ({before} -> {after}).");
    }
}
