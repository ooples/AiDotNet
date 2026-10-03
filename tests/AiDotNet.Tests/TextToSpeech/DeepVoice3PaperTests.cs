using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.Classic;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// Deep Voice 3 trains and synthesizes as its paper specifies (Ping et al. 2018): gated convolution blocks, an encoder
/// producing attention keys and values, a causal decoder with positional dot-product attention emitting r frames and a
/// done flag, and a converter predicting the linear spectrogram.
/// </summary>
/// <remarks>
/// Before this change Deep Voice 3 had no attention, no convolution blocks and no converter: its synthesis derived
/// frame counts from a fixed formula, and threw on every call because its encoder boundary pointed past the layer stack.
/// </remarks>
public class DeepVoice3PaperTests
{
    private const int MelBins = 8;

    private static DeepVoice3<double> CreateModel() => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 8, outputSize: MelBins) { RandomSeed = 11 },
        new DeepVoice3Options
        {
            SampleRate = 16000, HopSize = 200, FftSize = 64, WindowSize = 64, MelChannels = MelBins, OutputsPerStep = 2,
            EmbeddingDim = 16, EncoderChannels = 8, NumEncoderLayers = 2, DecoderPrenetSizes = new[] { 8, 16 },
            NumDecoderLayers = 2, AttentionDim = 8, NumConverterLayers = 1, ConverterChannels = 16, DropoutRate = 0.0,
            KeyPositionRate = 2.0, MaxDecoderSteps = 6, MinDecoderSteps = 2, LearningRate = 2e-3,
        });

    private static TtsTrainingSample<double> Sample()
    {
        const int frames = 9, bins = 33;
        var tokens = new Tensor<double>(new[] { 5 });
        for (int i = 0; i < 5; i++) tokens[i] = 10 + (i * 7) % 60;
        var mel = new Tensor<double>(new[] { frames, MelBins });
        var linear = new Tensor<double>(new[] { frames, bins });
        for (int f = 0; f < frames; f++)
        {
            for (int c = 0; c < MelBins; c++) mel[f, c] = Math.Sin(0.4 * f + 0.3 * c);
            for (int k = 0; k < bins; k++) linear[f, k] = Math.Cos(0.3 * f + 0.1 * k) - 1.0;
        }
        return new TtsTrainingSample<double> { Tokens = tokens, Mel = mel, LinearSpectrogram = linear };
    }

    [Fact(Timeout = 60000)]
    public async Task CausalBlock_NeverSeesTheFuture()
    {
        await Task.Yield();
        var block = new GatedConvolutionBlockLayer<double>(4, 4, 5, 1, causal: true, dropoutRate: 0.0, residual: false);
        var x = new Tensor<double>(new[] { 7, 4 });
        for (int i = 0; i < x.Length; i++) x[i] = Math.Sin(0.5 * i);
        var y1 = block.Forward(x);
        var changed = new Tensor<double>(x._shape, x.ToVector());
        for (int c = 0; c < 4; c++) changed[6, c] += 5.0;
        var y2 = block.Forward(changed);
        for (int t = 0; t < 6; t++)
            for (int c = 0; c < 4; c++) Assert.Equal(y1[t, c], y2[t, c], 12);
    }

    [Fact(Timeout = 180000)]
    public async Task Training_ReducesTheMelDoneAndLinearObjective()
    {
        await Task.Yield();
        var model = CreateModel();
        var sample = Sample();
        double before = model.EvaluateTrainingObjective(sample);
        for (int i = 0; i < 30; i++) model.Train(sample);
        double after = model.EvaluateTrainingObjective(sample);
        Assert.True(after < before, $"Objective did not fall ({before} -> {after}).");
    }

    [Fact(Timeout = 120000)]
    public async Task Synthesis_DecodesGroupsOfRFrames_AndAConverterSpectrogram()
    {
        await Task.Yield();
        var model = CreateModel();
        var mel = model.Synthesize("hello");
        Assert.Equal(2, mel.Rank);
        Assert.Equal(MelBins, mel.Shape[1]);
        Assert.Equal(0, mel.Shape[0] % 2);
        Assert.InRange(mel.Shape[0], 2 * 2, 6 * 2);
        var linear = model.PredictLinearSpectrogram("hello");
        Assert.Equal(mel.Shape[0], linear.Shape[0]);
        Assert.Equal(33, linear.Shape[1]);
        for (int i = 0; i < mel.Length; i++) Assert.True(double.IsFinite(mel[i]), $"mel[{i}] = {mel[i]}");
    }

    [Fact(Timeout = 60000)]
    public async Task TokenMelTraining_Throws_BecauseTheConverterTrainsOnTheLinearSpectrogram()
    {
        await Task.Yield();
        Assert.Throws<NotSupportedException>(() => CreateModel().Train(Sample().Tokens, Sample().Mel!));
    }
}
