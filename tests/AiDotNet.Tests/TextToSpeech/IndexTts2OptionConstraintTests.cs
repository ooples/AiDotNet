using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.Latest;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class IndexTts2OptionConstraintTests
{
    [Theory]
    [InlineData(16, 32)]
    [InlineData(32, 16)]
    public void ModelRejectsUnpairedDitAndWavenetWidths(int dit, int wavenet)
    {
        var error = Assert.Throws<ArgumentException>(() => new IndexTTS2<double>(
            new NeuralNetworkArchitecture<double>(inputFeatures: 1, outputSize: 5),
            new IndexTTS2Options { DitHidden = dit, WavenetHidden = wavenet }));
        Assert.Contains("WavenetHidden must equal DitHidden", error.Message);
    }

    [Fact]
    public void PairedWidthsResizeAndRunTheActualVelocityEstimator()
    {
        long Run(int width)
        {
            var layers = new List<LayerBase<double>>();
            var configuration = new IndexTtsSemanticToMelConfiguration(
                5, width, 2, 1, 6, 12, 8, 1, width, 2, 5, 0, 0, 16, 10, 7);
            var estimator = new IndexTtsDiT<double>(AiDotNet.Tensors.Engines.AiDotNetEngine.Current,
                layers, configuration, new Random(1337));
            var output = estimator.Forward(new Tensor<double>(new[] { 3, 5 }),
                new Tensor<double>(new[] { 3, 5 }), 0.37, new Tensor<double>(new[] { 6 }),
                new Tensor<double>(new[] { 3, 8 }), false);
            Assert.Equal(new[] { 3, 5 }, output.Shape);
            Assert.All(output, value => Assert.True(double.IsFinite(value)));
            return layers.Sum(layer => (long)layer.GetParameters().Length);
        }
        Assert.True(Run(32) > Run(16));
    }
}
