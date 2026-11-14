#if NET5_0_OR_GREATER
using System.Reflection;
using System.Runtime.CompilerServices;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

public partial class Elementwise01NumericalRegressionTests
{
    [Theory]
    [InlineData(0.0)]
    [InlineData(60000.0)]
    public void ImageBindHalfAudioEnergy_PreservesFiniteLogFloorAndWideSquares(double amplitude)
    {
        var model = NumericModel<Half, ImageBindNeuralNetwork<Half>>();
        var waveform = new Tensor<Half>(Enumerable.Repeat((Half)amplitude, 320).ToArray(), new[] { 320 });
        var result = Assert.IsType<Tensor<Half>>(Invoke(typeof(ImageBindNeuralNetwork<Half>), model,
            "ComputeMelSpectrogram", waveform, 16000));
        Assert.Equal(new[] { 128, 2 }, result.Shape);
        double energy = Math.Log(amplitude * amplitude + 1e-10);
        for (int mel = 0; mel < 128; mel++)
        for (int frame = 0; frame < 2; frame++)
            Assert.Equal((Half)((1 - mel / 128.0 * 0.5) * energy), result[mel, frame]);
    }

    [Fact]
    public void BertHalfPaddingPositions_ContinueCountingPast2048()
    {
        const int length = 2052, positions = length + 1;
        using var layer = new BertEmbeddingLayer<Half>(2, 2, positions,
            dropoutProbability: 0, layerNormEpsilon: 0.001, paddingTokenId: 0);
        layer.SetTrainingMode(false);
        var ids = new Tensor<Half>(Enumerable.Repeat((Half)1, length).ToArray(), new[] { 1, length });
        _ = layer.Forward(ids);
        layer.WordEmbedding.SetParameters(new Vector<Half>(4));
        layer.TokenTypeEmbedding.SetParameters(new Vector<Half>(4));
        var table = new Vector<Half>(positions * 2);
        for (int row = 0; row < positions; row++)
        {
            table[row * 2] = (Half)(-2);
            table[row * 2 + 1] = (Half)2;
        }
        table[length * 2] = (Half)2;
        table[length * 2 + 1] = (Half)(-2);
        layer.PositionEmbedding.SetParameters(table);
        var output = layer.Forward(ids);
        Assert.InRange((double)output[0, length - 1, 0], 0.99, 1.01);
        Assert.InRange((double)output[0, length - 1, 1], -1.01, -0.99);
        Assert.InRange((double)output[0, 2047, 0], -1.01, -0.99);
    }

    // Sinusoidal tables and per-head constants are built once in double and rounded to T, as the
    // per-element double math was: an angle carried in float drifts by tens of ulps by position 500.
    private const float TableTolerance = 2.5e-7f;   // two roundings of a value in [-1, 1] to float

    private static double PairedAngle(int position, int column, int width)
        => position / Math.Pow(10000, (double)(column - (column % 2)) / width);

    private static void AssertPairedTable(Func<int, int, float> actual, int positions, int width)
    {
        for (int p = 0; p < positions; p++)
        for (int c = 0; c < width; c++)
        {
            double angle = PairedAngle(p, c, width);
            float expected = (float)(c % 2 == 0 ? Math.Sin(angle) : Math.Cos(angle));
            Assert.True(Math.Abs(expected - actual(p, c)) <= TableTolerance,
                $"[{p}, {c}]: expected {expected}, got {actual(p, c)}");
        }
    }

    [Fact]
    public void ALiBiSlopes_AreExactPowersOfTwo()
    {
        using var single = new ALiBiPositionalBiasLayer<float>(numHeads: 12, maxSequenceLength: 4);
        var slopes = single.GetSlopes();
        for (int h = 0; h < 12; h++)
            Assert.Equal((float)Math.Pow(2.0, -8.0 / 12 * (h + 1)), slopes[h]);

        using var half = new ALiBiPositionalBiasLayer<Half>(numHeads: 3, maxSequenceLength: 4);
        var halfSlopes = half.GetSlopes();
        for (int h = 0; h < 3; h++)
            Assert.Equal((Half)Math.Pow(2.0, -8.0 / 3 * (h + 1)), halfSlopes[h]);
    }

    [Fact]
    public void AudioVisualSinusoidalTable_MatchesDoubleReferenceAtLongPositions()
    {
        const int width = 16, length = 500;
        var model = NumericModel<float, AudioVisualCorrespondenceNetwork<float>>();
        (typeof(AudioVisualCorrespondenceNetwork<float>).GetField("_embeddingDimension", BindingFlags.NonPublic | BindingFlags.Instance)
            ?? throw new InvalidOperationException("Missing _embeddingDimension")).SetValue(model, width);
        var table = Assert.IsType<Tensor<float>>(Invoke(typeof(AudioVisualCorrespondenceNetwork<float>), model, "SinusoidalEmbedding", length));
        for (int p = 0; p < length; p++)
        for (int i = 0; i < width; i++)
        {
            double angle = p * Math.Exp(i * -Math.Log(10000.0) / width);
            float expected = (float)(i % 2 == 0 ? Math.Sin(angle) : Math.Cos(angle));
            Assert.True(Math.Abs(expected - table[p, i]) <= TableTolerance, $"[{p}, {i}]: expected {expected}, got {table[p, i]}");
        }
    }

    [Fact]
    public void FlamingoPositionalTable_MatchesDoubleReferenceAtLongPositions()
    {
        const int width = 16, length = 600;
        var model = NumericModel<float, FlamingoNeuralNetwork<float>>();
        var table = new Tensor<float>(new[] { length, width });
        Invoke(typeof(FlamingoNeuralNetwork<float>), model, "InitializePositionalEmbeddings", table);
        AssertPairedTable((p, c) => table[p, c], length, width);
    }

    [Fact]
    public void BidirectionalClozePositionalTable_MatchesDoubleReferenceAtLongPositions()
    {
        const int width = 16, length = 600;
        var layer = (BidirectionalClozeNetworkLayer<float>)RuntimeHelpers.GetUninitializedObject(typeof(BidirectionalClozeNetworkLayer<float>));
        var table = Assert.IsType<Tensor<float>>(Invoke(typeof(BidirectionalClozeNetworkLayer<float>), layer, "PositionalEncoding", length, width));
        AssertPairedTable((p, c) => table[p, c], length, width);
    }

    [Fact]
    public void DocFormerSpatialEmbedding_WithZeroTables_IsTheDoubleReferencePositionTable()
    {
        const int width = 16, length = 600;
        using var layer = new DocFormerSpatialEmbeddingLayer<float>(width, maxPosition2D: 8);
        var boxes = new Tensor<float>(new[] { length, 4 });
        _ = layer.Forward(boxes);
        layer.SetParameters(new Vector<float>(layer.GetParameters().Length));
        var output = layer.Forward(boxes);
        AssertPairedTable((p, c) => output[p, c], length, width);
    }

    [Fact]
    public void ContinuumDefaultLearningRates_AreBaseRateOverPowersOfTen()
    {
        var layer = (ContinuumMemorySystemLayer<float>)RuntimeHelpers.GetUninitializedObject(typeof(ContinuumMemorySystemLayer<float>));
        var rates = Assert.IsType<float[]>(Invoke(typeof(ContinuumMemorySystemLayer<float>), layer, "CreateDefaultLearningRates", 6));
        Assert.Equal(6, rates.Length);
        for (int i = 0; i < 6; i++)
            Assert.Equal((float)(0.01 / Math.Pow(10, i)), rates[i]);
    }
}
#endif
