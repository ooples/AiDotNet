using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.Vocoders;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class HiFiGanSamplingSeedTests
{
    private static readonly string[] Fields = { "_direction", "_gain", "_bias", "_u", "_v" };

    [Fact]
    public void SamplingSeedControlsEveryGeneratorDiscriminatorParameterAndSpectralBuffer()
    {
        var first = Snapshot(Create(1234));
        var second = Snapshot(Create(1234));
        Assert.Equal(first.Keys, second.Keys);
        foreach (var pair in first) Assert.Equal(pair.Value, second[pair.Key]);
        var different = Snapshot(Create(1235));
        Assert.NotEqual(first["generator/0/_direction"], different["generator/0/_direction"]);
        Assert.NotEqual(first["generator/0/_bias"], different["generator/0/_bias"]);
        Assert.NotEqual(first["discriminator/0/_direction"], different["discriminator/0/_direction"]);
        Assert.Contains(first, pair => pair.Key.StartsWith("discriminator/") && pair.Key.EndsWith("/_u")
            && !pair.Value.SequenceEqual(different[pair.Key]));
        Assert.Contains(first, pair => pair.Key.StartsWith("discriminator/") && pair.Key.EndsWith("/_v")
            && !pair.Value.SequenceEqual(different[pair.Key]));
    }

    [Fact]
    public void DefaultInitializationStreamDoesNotChangePaperNormalDrawSequence()
    {
        HiFiGanGenerator<double> Generator(int defaults) => new(new CpuEngine(), 2, 8, new[] { 2 }, new[] { 4 },
            new[] { 3 }, new[] { new[] { 1, 3 } }, true,
            initialization: new Random(1241), defaultInitialization: new Random(defaults));
        var first = Generator(1245).Layers.Cast<NormedConv1DLayer<double>>().ToArray();
        var second = Generator(9876).Layers.Cast<NormedConv1DLayer<double>>().ToArray();
        Assert.NotEqual(Tensor(first[0], "_direction"), Tensor(second[0], "_direction"));
        for (int layer = 1; layer < first.Length; layer++)
        {
            Assert.Equal(Tensor(first[layer], "_direction"), Tensor(second[layer], "_direction"));
            Assert.Equal(Tensor(first[layer], "_gain"), Tensor(second[layer], "_gain"));
        }
        Assert.Contains(Enumerable.Range(0, first.Length), layer =>
            !Tensor(first[layer], "_bias").SequenceEqual(Tensor(second[layer], "_bias")));
    }

    [Fact]
    public void ExplicitDefaultDrawsRetainUniformBoundsAndUnitSpectralVectors()
    {
        var layer = new NormedConv1DLayer<double>(3, 4, 5, 1, 1, 1, 2, false,
            ConvolutionNormalization.Spectral, outputPadding: 0, useBias: true, initialization: new Random(77));
        double bound = 1 / Math.Sqrt(15);
        Assert.All(Tensor(layer, "_direction"), value => Assert.InRange(value, -bound, bound));
        Assert.All(Tensor(layer, "_bias"), value => Assert.InRange(value, -bound, bound));
        foreach (string field in new[] { "_u", "_v" })
            Assert.InRange(Math.Abs(Tensor(layer, field).Sum(v => v * v) - 1), 0, 1e-12);
    }

    private static HiFiGAN<double> Create(int seed) => new(
        new NeuralNetworkArchitecture<double>(inputFeatures: 2, outputSize: 1),
        new HiFiGANOptions
        {
            SamplingSeed = seed, MelChannels = 2, UpsampleInitialChannels = 8,
            UpsampleRates = new[] { 2 }, UpsampleKernelSizes = new[] { 4 },
            ResblockKernelSizes = new[] { 3 }, ResblockDilationSizes = new[] { new[] { 1 } },
            DiscriminatorPeriods = new[] { 2 }, DiscriminatorWidthDivisor = 128,
            FftSize = 8, WindowSize = 8, HopSize = 2, SegmentSize = 16,
        });

    private static Dictionary<string, double[]> Snapshot(HiFiGAN<double> model)
    {
        var result = new Dictionary<string, double[]>();
        foreach (var item in new[] { ("generator", "_generator"), ("discriminator", "_discriminators") })
        {
            var module = typeof(HiFiGAN<double>).GetField(item.Item2, BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(model)!;
            var layers = (IReadOnlyList<LayerBase<double>>)module.GetType().GetProperty("Layers")!.GetValue(module)!;
            for (int i = 0; i < layers.Count; i++)
                foreach (string field in Fields) result.Add($"{item.Item1}/{i}/{field}", Tensor((NormedConv1DLayer<double>)layers[i], field));
        }
        return result;
    }

    private static double[] Tensor(NormedConv1DLayer<double> layer, string field) =>
        ((Tensor<double>)typeof(NormedConv1DLayer<double>).GetField(field, BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(layer)!).ToArray();
}
