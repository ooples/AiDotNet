using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// The lazy shape walk (NeuralNetworkBase.ResolveLazyLayerShapes) must resolve each layer against the shape the
/// previous layer actually produces.
/// </summary>
public class LazyShapeWalkTests
{
    /// <summary>
    /// A lazy LayerNorm declares its output as a fully unknown [-1]. Filling that from the walk's last rank-2 shape
    /// (the input [batch, 16]) told the next dense layer it took 16 features where the dense layer before the norm
    /// produced 32, and Predict then failed in the matmul, which is how DAC, FastConformer and a dozen other audio
    /// models broke. The walk must pass the dense layer's [32] through the norm unchanged.
    /// </summary>
    [Fact(Timeout = 60000)]
    public async Task FullyDynamicDeclaration_PassesTheRunningShapeThrough()
    {
        await Task.Yield();
        var gelu = (IActivationFunction<double>)new GELUActivation<double>();
        var identity = (IActivationFunction<double>)new IdentityActivation<double>();
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple, inputSize: 16, outputSize: 4,
            layers: new List<ILayer<double>>
            {
                new FullyConnectedLayer<double>(32, gelu),
                new LayerNormalizationLayer<double>(),
                new FullyConnectedLayer<double>(64, gelu),
                new LayerNormalizationLayer<double>(),
                new FullyConnectedLayer<double>(4, identity),
            });
        using var network = new FeedForwardNeuralNetwork<double>(architecture);

        var input = new Tensor<double>([2, 16]);
        for (int i = 0; i < input.Length; i++) input[i] = System.Math.Sin(0.37 * i);
        var output = network.Predict(input);

        Assert.Equal(new[] { 2, 4 }, output.Shape.ToArray());
        Assert.All(output.ToArray(), v => Assert.True(double.IsFinite(v)));
    }
}
