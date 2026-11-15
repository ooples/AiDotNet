using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ComputerVision.Segmentation.Salient;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision.Segmentation;

/// <summary>
/// U²-Net (Qin et al. 2020, <c>model/u2net.py</c>) and rembg's u2net pre-processing.
/// </summary>
public class U2NetTests
{
    /// <summary>Every middle width 2, every output width 4, RGB in, one map out.</summary>
    private static U2NetConfiguration Tiny() => new U2NetConfiguration
    {
        EncoderMiddle = new[] { 2, 2, 2, 2, 2, 2 },
        EncoderOutput = new[] { 4, 4, 4, 4, 4, 4 },
        DecoderMiddle = new[] { 2, 2, 2, 2, 2 },
        DecoderOutput = new[] { 4, 4, 4, 4, 4 },
    };

    private static Tensor<double> Image(int channels, int height, int width, int seed)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        var values = new double[channels * height * width];
        for (int i = 0; i < values.Length; i++) values[i] = random.NextDouble();
        return new Tensor<double>(new[] { 1, channels, height, width }, new Vector<double>(values));
    }

    [Fact]
    public void ParameterCount_OfATinyConfiguration_IsTheHandDerivedCount()
    {
        // REBNCONV(i -> o): conv weight o*i*9 + conv bias o + BN weight o + BN bias o = 9oi + 3o.
        // RSU-L(in, m=2, o=4): rebnconvin(in -> 4) = 36 in + 12, rebnconv1(4 -> 2) = 78,
        // rebnconv2..L(2 -> 2) = 42 (L - 1), rebnconv(L-1)d..2d(4 -> 2) = 78 (L - 2), rebnconv1d(4 -> 4) = 156,
        // so RSU-L = 36 in + 48 + 120 L (RSU-4F has the same layers as RSU-4).
        // Encoder: RSU-7(3) 996 + RSU-6(4) 912 + RSU-5(4) 792 + RSU-4(4) 672 + 2 x RSU-4F(4) 672 = 4716.
        // Decoder (inputs 4 + 4 = 8): RSU-4F 816 + RSU-4 816 + RSU-5 936 + RSU-6 1056 + RSU-7 1176 = 4800.
        // side1..6: 6 x (1*4*9 + 1) = 222; outconv: 1*6 + 1 = 7. Total 4716 + 4800 + 222 + 7 = 9745.
        var network = new U2Net<double>(Tiny(), seed: 1);
        Assert.Equal(9745L, network.ParameterCount);
        Assert.Equal(9745, network.GetParameters().Length);
    }

    [Fact]
    public void ParameterCount_OfU2NetSmall_MatchesThePublishedU2NetP()
    {
        // u2netp.pth: sum(p.numel() for p in U2NETP().parameters()) = 1,131,181.
        var network = new U2Net<float>(U2NetConfiguration.Small(), seed: 1);
        Assert.Equal(1131181L, network.ParameterCount);
    }

    [Theory]
    [InlineData(7, false)]
    [InlineData(5, false)]
    [InlineData(4, true)]
    public void ResidualUBlock_KeepsTheSpatialSizeAndMapsToTheOutputWidth(int height, bool dilated)
    {
        var block = new ResidualUBlock<double>(height, dilated, inputs: 3, middle: 2, outputs: 5, RandomHelper.CreateSeededRandom(4));
        Assert.Equal(height, block.Encoder.Count);
        Assert.Equal(height - 1, block.Decoder.Count);
        // Odd sizes exercise MaxPool2d(ceil_mode=True) and the bilinear upsample back to the skip size.
        var output = block.Forward(Image(3, 13, 11, 2), 1e-5);
        Assert.Equal(new[] { 1, 5, 13, 11 }, output.Shape.ToArray());
    }

    [Fact]
    public void ResidualUBlock_UsesTheOfficialDilations()
    {
        var rsu = new ResidualUBlock<double>(5, false, 3, 2, 4, RandomHelper.CreateSeededRandom(1));
        Assert.Equal(new[] { 1, 1, 1, 1, 2 }, rsu.Encoder.Select(c => c.Dilation).ToArray());
        Assert.Equal(new[] { 1, 1, 1, 1 }, rsu.Decoder.Select(c => c.Dilation).ToArray());
        var rsu4f = new ResidualUBlock<double>(4, true, 3, 2, 4, RandomHelper.CreateSeededRandom(1));
        Assert.Equal(new[] { 1, 2, 4, 8 }, rsu4f.Encoder.Select(c => c.Dilation).ToArray());
        Assert.Equal(new[] { 4, 2, 1 }, rsu4f.Decoder.Select(c => c.Dilation).ToArray());
        Assert.Equal(2 * 2, rsu4f.Decoder[0].Inputs);
        Assert.Equal(4, rsu4f.Decoder[2].Outputs);
    }

    [Fact]
    public void Network_StageShapes_FollowTheReference()
    {
        var network = new U2Net<double>(Tiny(), seed: 2);
        Assert.Equal(new[] { 7, 6, 5, 4, 4, 4 }, network.EncoderStages.Select(s => s.Height).ToArray());
        Assert.Equal(new[] { false, false, false, false, true, true }, network.EncoderStages.Select(s => s.Dilated).ToArray());
        Assert.Equal(new[] { 4, 4, 5, 6, 7 }, network.DecoderStages.Select(s => s.Height).ToArray());
        Assert.Equal(new[] { true, false, false, false, false }, network.DecoderStages.Select(s => s.Dilated).ToArray());
        Assert.All(network.DecoderStages, s => Assert.Equal(8, s.Input.Inputs));

        var outputs = network.Forward(Image(3, 19, 23, 5));
        Assert.Equal(7, outputs.Count);
        Assert.All(outputs, o => Assert.Equal(new[] { 1, 1, 19, 23 }, o.Shape.ToArray()));
    }

    [Fact]
    public void Forward_IsSeededDeterministic_AndOutputsProbabilities()
    {
        var input = Image(3, 16, 12, 9);
        var first = new U2Net<double>(Tiny(), seed: 11).Forward(input);
        var second = new U2Net<double>(Tiny(), seed: 11).Forward(input);
        var other = new U2Net<double>(Tiny(), seed: 12).Forward(input);
        for (int k = 0; k < first.Count; k++)
        {
            var a = first[k].ToArray();
            Assert.Equal(a, second[k].ToArray());
            Assert.All(a, v => Assert.InRange(v, 0.0, 1.0));
            Assert.All(a, v => Assert.False(double.IsNaN(v)));
        }

        Assert.NotEqual(first[0].ToArray(), other[0].ToArray());
    }

    [Fact]
    public void LoadTorchWeights_ReadsTheOfficialStateDictNames()
    {
        var network = new U2Net<double>(Tiny(), seed: 3);
        var requested = new Dictionary<string, int[]>();
        network.LoadTorchWeights((name, shape) =>
        {
            requested.Add(name, shape);
            return new double[shape.Aggregate(1, (a, b) => a * b)];
        });

        // 11 RSU blocks: 2L conv+BN layers each, 6 tensors per layer; 6 sides and outconv with weight and bias.
        int layers = new[] { 7, 6, 5, 4, 4, 4, 4, 4, 5, 6, 7 }.Sum(l => 2 * l);
        Assert.Equal((6 * layers) + 14, requested.Count);
        Assert.Equal(new[] { 4, 3, 3, 3 }, requested["stage1.rebnconvin.conv_s1.weight"]);
        Assert.Equal(new[] { 2, 2, 3, 3 }, requested["stage1.rebnconv7.conv_s1.weight"]);
        Assert.Equal(new[] { 2 }, requested["stage6.rebnconv4.bn_s1.running_var"]);
        Assert.Equal(new[] { 4, 4, 3, 3 }, requested["stage1d.rebnconv1d.conv_s1.weight"]);
        Assert.Equal(new[] { 4, 8, 3, 3 }, requested["stage5d.rebnconvin.conv_s1.weight"]);
        Assert.Equal(new[] { 1, 4, 3, 3 }, requested["side6.weight"]);
        Assert.Equal(new[] { 1, 6, 1, 1 }, requested["outconv.weight"]);
        Assert.True(requested.ContainsKey("outconv.bias"));
        Assert.All(network.GetParameters().ToArray(), v => Assert.Equal(0.0, v));
    }

    [Fact]
    public void Segmenter_ReturnsAnImageSizedAlphaInTheUnitRange()
    {
        var segmenter = new U2NetForegroundSegmenter<double>(new U2Net<double>(Tiny(), seed: 4), inputSize: 16);
        var image = Image(3, 10, 14, 6);
        var alpha = segmenter.Segment(image);
        Assert.Equal(new[] { 10, 14 }, alpha.Shape.ToArray());
        var values = alpha.ToArray();
        Assert.All(values, v => Assert.InRange(v, 0.0, 1.0));
        // Pillow stores the resized matte as 8-bit.
        Assert.All(values, v => Assert.Equal(Math.Round(v * 255.0), v * 255.0, 9));
        Assert.Equal(values, segmenter.Segment(image).ToArray());
    }

    [Fact]
    public void Lanczos_MatchesPillowsCoefficients()
    {
        // Pillow, 4 -> 2: scale 2, support 6, centre 1 covers inputs 0..3 with Lanczos3((k - 1 + 0.5) / 2).
        var (start, weights) = LanczosResampler.Coefficients(4, 2);
        Assert.Equal(0, start[0]);
        Assert.Equal(4, weights[0].Length);
        double Sinc(double x) => x == 0 ? 1.0 : Math.Sin(Math.PI * x) / (Math.PI * x);
        var raw = Enumerable.Range(0, 4).Select(k => { double x = (k - 1 + 0.5) / 2.0; return Sinc(x) * Sinc(x / 3.0); }).ToArray();
        double total = raw.Sum();
        for (int k = 0; k < 4; k++)
            Assert.Equal(raw[k] / total, weights[0][k], 12);
        // Identity size keeps a constant image.
        var same = LanczosResampler.Resize(Enumerable.Repeat(0.25, 12).ToArray(), 1, 3, 4, 3, 4, 0.0, 1.0);
        Assert.All(same, v => Assert.Equal(0.25, v, 12));
    }
}
