using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class NormedConvParameterOwnershipTests
{
    [Theory]
    [InlineData(ConvolutionNormalization.None, false, 1)]
    [InlineData(ConvolutionNormalization.None, true, 2)]
    [InlineData(ConvolutionNormalization.Weight, false, 2)]
    [InlineData(ConvolutionNormalization.Weight, true, 3)]
    [InlineData(ConvolutionNormalization.Spectral, false, 1)]
    [InlineData(ConvolutionNormalization.Spectral, true, 2)]
    public void OptimizerSurface_ContainsOnlyParametersUsedByTheConfiguredConvolution(
        ConvolutionNormalization normalization, bool bias, int expectedCount)
    {
        var layer = new NormedConv1DLayer<double>(2, 3, 3, 1, 1, 1, 1, false, normalization, bias);
        layer.SetTrainingMode(false);
        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(new List<ILayer<double>> { layer });
        Assert.Equal(expectedCount, parameters.Count);
        int learned = 18 + (normalization == ConvolutionNormalization.Weight ? 3 : 0) + (bias ? 3 : 0);
        Assert.Equal(learned, parameters.Sum(parameter => parameter.Length));
        // This codebase's flat checkpoint surface also contains persistent buffers:
        // the three-element spectral u and six-element v are state, not optimizer parameters.
        Assert.Equal(learned + 9, layer.ParameterCount);
        Assert.Equal(layer.ParameterCount, layer.GetParameters().Length);
        var input = new Tensor<double>(new[] { 1, 2, 7 }, new Vector<double>(
            Enumerable.Range(0, 14).Select(i => Math.Sin(i + .2)).ToArray()));
        using var tape = new GradientTape<double>();
        var output = layer.Forward(input);
        var loss = AiDotNetEngine.Current.ReduceMean(AiDotNetEngine.Current.TensorMultiply(output, output), new[] { 0, 1, 2 }, false);
        var gradients = tape.ComputeGradients(loss, sources: parameters);
        foreach (var parameter in parameters)
        {
            Assert.True(gradients.ContainsKey(parameter), "The optimizer owns a tensor that the configured convolution never uses.");
            Assert.All(gradients[parameter].ToArray(), value => Assert.True(TtsPaperNumerics.IsFinite(value)));
        }
    }
}
