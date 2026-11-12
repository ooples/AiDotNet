#if NET5_0_OR_GREATER
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
}
#endif
