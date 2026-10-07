using System;
using System.Linq;
using AiDotNet.Audio.Enhancement;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Audio;

/// <summary>
/// Conv-TasNet trains end to end (#2298). It used to keep raw weight tensors with a hand-written update that
/// only nudged decoder slots by (prediction - target) * 0.01: the encoder, the mask head and every TCN block
/// never changed, and the SI-SNR loss was computed and discarded.
/// </summary>
public class ConvTasNetTrainingTests
{
    private const int Samples = 128;

    private static ConvTasNet<double> CreateModel(int tcnKernelSize = 3) => new(
        new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputSize: Samples, outputSize: Samples),
        new ConvTasNetOptions
        {
            SampleRate = 8000, EncoderDim = 16, KernelSize = 8, BottleneckDim = 8, HiddenDim = 16,
            NumBlocks = 2, NumRepeats = 1, TcnKernelSize = tcnKernelSize, NumSources = 2
        });

    private static (Tensor<double> Mixture, Tensor<double> Sources) TwoTones(int seed)
    {
        // Two sinusoids at different frequencies, mixed: a separable target with a known answer.
        var rng = RandomHelper.CreateSeededRandom(seed);
        double f1 = 0.05 + 0.02 * rng.NextDouble(), f2 = 0.21 + 0.02 * rng.NextDouble();
        var sources = new Tensor<double>(new[] { 1, 2, Samples });
        var mixture = new Tensor<double>(new[] { 1, Samples });
        for (int t = 0; t < Samples; t++)
        {
            double a = Math.Sin(2 * Math.PI * f1 * t), b = 0.6 * Math.Sin(2 * Math.PI * f2 * t + 1.0);
            sources[0, 0, t] = a;
            sources[0, 1, t] = b;
            mixture[0, t] = a + b;
        }

        return (mixture, sources);
    }

    [Fact]
    public void OneStep_MovesEveryPartOfTheNetwork()
    {
        var model = CreateModel();
        var (mixture, sources) = TwoTones(1);
        model.Predict(mixture);
        var before = model.Layers.Select(layer => layer.GetParameters().ToArray()).ToList();

        model.Train(mixture, sources);

        // Every layer that has weights must have moved: the encoder, the bottleneck, each TCN block's 1x1,
        // PReLU, gLN, depthwise, residual and skip layers, the mask head and the decoder.
        for (int i = 0; i < model.Layers.Count; i++)
        {
            var after = model.Layers[i].GetParameters().ToArray();
            if (before[i].Length == 0) continue;
            Assert.All(after, value => Assert.False(double.IsNaN(value) || double.IsInfinity(value)));
            Assert.True(before[i].Zip(after, (a, b) => a != b).Any(changed => changed),
                $"Layer {i} ({model.Layers[i].GetType().Name}) did not change after a training step.");
        }

        Assert.Contains(model.Layers, layer => layer is Conv1DTransposeLayer<double>);
        Assert.True(model.Layers.OfType<Conv1DLayer<double>>().Count() >= 6,
            "The TCN separator was not built from convolution layers.");
    }

    [Fact]
    public void Training_RaisesSiSnrOnAFixedMixture()
    {
        var model = CreateModel();
        var (mixture, sources) = TwoTones(2);
        var loss = new AiDotNet.LossFunctions.NegativeSiSnrLoss<double>();
        double Loss() => loss.ComputeTapeLoss(model.Predict(mixture), sources)[0];

        double before = Loss();
        for (int step = 0; step < 40; step++) model.Train(mixture, sources);
        double after = Loss();

        Assert.True(after < before - 1.0,
            $"Forty steps on one mixture should improve SI-SNR by over 1 dB; negative SI-SNR went {before:F3} -> {after:F3}.");
    }

    [Fact]
    public void Output_IsOneWaveformPerSourceAtTheInputLength()
    {
        var model = CreateModel();
        var (mixture, _) = TwoTones(3);

        var separated = model.Predict(mixture);

        Assert.Equal(new[] { 1, 2, Samples }, separated.Shape.ToArray());
        Assert.All(separated.ToArray(), value => Assert.False(double.IsNaN(value) || double.IsInfinity(value)));
    }

    [Theory]
    [InlineData(130)]
    [InlineData(5)]
    public void Output_KeepsALengthTheFramesDoNotTile_AndEachBatchRowIsIndependent(int length)
    {
        // 128 samples tile the stride exactly; these lengths take the pad-then-crop path (5 is shorter
        // than the 8-sample encoder kernel). gLN normalizes each example on its own, so a row's output
        // must not depend on what else is in the batch.
        var model = CreateModel();
        var rng = RandomHelper.CreateSeededRandom(length);
        var batch = new Tensor<double>(new[] { 2, length });
        var rows = new[] { new Tensor<double>(new[] { 1, length }), new Tensor<double>(new[] { 1, length }) };
        for (int r = 0; r < 2; r++)
        {
            for (int t = 0; t < length; t++)
            {
                double v = rng.NextDouble() * 2 - 1;
                batch[r, t] = v;
                rows[r][0, t] = v;
            }
        }

        var together = model.Predict(batch);
        Assert.Equal(new[] { 2, 2, length }, together.Shape.ToArray());

        for (int r = 0; r < 2; r++)
        {
            var alone = model.Predict(rows[r]);
            Assert.Equal(new[] { 1, 2, length }, alone.Shape.ToArray());
            for (int c = 0; c < 2; c++)
                for (int t = 0; t < length; t++)
                    Assert.Equal(alone[0, c, t], together[r, c, t], 10);
        }
    }

    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    public void EvenTcnKernel_IsRejectedAtConstruction(int tcnKernelSize)
    {
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() => CreateModel(tcnKernelSize));
        Assert.Equal("TcnKernelSize", ex.ParamName);
    }

    [Fact]
    public void EnhancementStrength_ScalesWhatPredictReturns()
    {
        var model = CreateModel();
        var (mixture, _) = TwoTones(4);
        var full = model.Predict(mixture).ToArray();

        model.EnhancementStrength = 0.5;
        var half = model.Predict(mixture).ToArray();

        Assert.Contains(full, value => Math.Abs(value) > 1e-9);
        for (int i = 0; i < full.Length; i++)
            Assert.Equal(full[i] * 0.5, half[i], 12);
    }
}