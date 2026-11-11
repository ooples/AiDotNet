using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.SpeechRecognition;

/// <summary>
/// <c>LayerHelper.CreateDefaultConformerLayers</c> builds the Conformer of Gulati et al. (2020): convolution
/// subsampling, then Conformer blocks with their convolution module. It used to build a pre-LN Transformer behind
/// a Dense stack that kept every frame, and 21 speech recognizers inherited that.
/// </summary>
public class ConformerFactoryTests
{
    [Fact]
    public void Factory_BuildsSubsamplingThenConformerBlocks()
    {
        var layers = LayerHelper<double>.CreateDefaultConformerLayers(
            encoderDim: 16, numLayers: 2, numAttentionHeads: 2, numMels: 16, vocabSize: 5, dropoutRate: 0.0).ToList();

        Assert.IsType<ConvSubsamplingLayer<double>>(layers[0]);
        Assert.Equal(2, layers.OfType<ConformerBlockLayer<double>>().Count());
        Assert.DoesNotContain(layers, layer => layer is TransformerEncoderBlock<double>);
    }

    [Theory]
    [InlineData(7, 1)]
    [InlineData(64, 15)]
    [InlineData(100, 24)]
    public void Subsampling_ShortensTimeFourTimes_AsTheReferenceDoes(int frames, int expected)
    {
        var layer = new ConvSubsamplingLayer<double>(inputFeatures: 16, modelDim: 8);

        var output = layer.Forward(Features(frames, 16));

        Assert.Equal(expected, ConvSubsamplingLayer<double>.Subsampled(frames));
        Assert.Equal(new[] { 1, expected, 8 }, output.Shape.ToArray());
    }

    [Fact]
    public void Subsampling_CountsTwoConvolutionsAndTheProjection()
    {
        const int mels = 16, dim = 8;
        var layer = new ConvSubsamplingLayer<double>(mels, dim);

        // conv1: 1 -> dim, 3x3, biased; conv2: dim -> dim, 3x3, biased; projection: dim * F' -> dim, biased.
        int subsampledMels = ConvSubsamplingLayer<double>.Subsampled(mels);
        long expected = (9L * dim + dim) + (9L * dim * dim + dim) + ((long)dim * subsampledMels * dim + dim);
        Assert.Equal(expected, layer.ParameterCount);
    }

    [Fact]
    public void Subsampling_RejectsInputsItCannotReduce()
    {
        var layer = new ConvSubsamplingLayer<double>(inputFeatures: 16, modelDim: 8);

        Assert.Throws<ArgumentException>(() => layer.Forward(Features(6, 16)));
        Assert.Throws<ArgumentException>(() => layer.Forward(Features(32, 12)));
    }

    [Fact]
    public void Subsampling_GradientReachesTheFirstConvolution()
    {
        var layer = new ConvSubsamplingLayer<double>(inputFeatures: 16, modelDim: 8);
        var input = Features(32, 16);

        using var tape = new GradientTape<double>();
        var output = layer.Forward(input);
        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(new[] { layer }, structureVersion: -1);
        var loss = AiDotNet.Tensors.Engines.AiDotNetEngine.Current.ReduceSum(
            AiDotNet.Tensors.Engines.AiDotNetEngine.Current.TensorMultiply(output, output), new[] { 0, 1, 2 }, keepDims: false);
        var gradients = tape.ComputeGradients(loss, parameters);

        // The first registered tensor is conv1's kernel: a tape break anywhere in the reshape/permute chain
        // between it and the projection leaves it without a gradient.
        var first = gradients[parameters[0]];
        Assert.NotNull(first);
        Assert.Contains(first!.ToArray(), value => Math.Abs(value) > 1e-12);
    }

    private static Tensor<double> Features(int frames, int mels)
    {
        var random = new Random(frames * 31 + mels);
        var tensor = new Tensor<double>([1, frames, mels]);
        for (int i = 0; i < tensor.Length; i++) tensor[i] = random.NextDouble() - 0.5;
        return tensor;
    }
}
