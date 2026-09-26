using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// Per-layer learning rates (PyTorch's param_groups): a layer's update is its own rate's, exactly, and the
/// rest of the model is untouched.
/// </summary>
public class LayerLearningRateGroupTests
{
    private static NeuralNetwork<double> Network()
    {
        var layers = new List<ILayer<double>>
        {
            new DenseLayer<double>(6, (IActivationFunction<double>)new TanhActivation<double>()),
            new DenseLayer<double>(2, (IActivationFunction<double>)new IdentityActivation<double>()),
        };
        var arch = new NeuralNetworkArchitecture<double>(
            InputType.OneDimensional, NeuralNetworkTaskType.Regression, NetworkComplexity.Simple,
            inputSize: 4, outputSize: 2, layers: layers) { RandomSeed = 7 };
        return new NeuralNetwork<double>(arch);
    }

    private static (Tensor<double> X, Tensor<double> Y) Data()
    {
        var x = new Tensor<double>(new[] { 3, 4 });
        var y = new Tensor<double>(new[] { 3, 2 });
        for (int i = 0; i < x.Length; i++) x[i] = Math.Sin(i + 1.0);
        for (int i = 0; i < y.Length; i++) y[i] = Math.Cos(i + 1.0);
        return (x, y);
    }

    private static double[][] Params(NeuralNetwork<double> n) =>
        n.Layers.OfType<LayerBase<double>>().Select(l => l.GetParameters().ToArray()).ToArray();

    [Fact]
    public void A_layer_scale_scales_exactly_that_layers_update_and_nothing_else()
    {
        var (x, y) = Data();
        // Built twice rather than copied (a DeepCopy shares the source's optimizer), then given the same weights.
        var reference = Network();
        var scaled = Network();
        reference.Predict(x);
        scaled.Predict(x);
        scaled.SetParameters(reference.GetParameters()); // same start, separate optimizers

        // Both take the eager per-group step: a policy whose factor is 1 on the reference, 0.5 on the other.
        ((LayerBase<double>)reference.Layers[0]).MaxLearningRate = 1e9;
        ((LayerBase<double>)scaled.Layers[0]).LearningRateScale = 0.5;

        var start = Params(reference);
        Assert.Equal(start.SelectMany(p => p), Params(scaled).SelectMany(p => p));

        reference.Train(x, y);
        scaled.Train(x, y);
        var afterReference = Params(reference);
        var afterScaled = Params(scaled);

        double moved = 0;
        for (int i = 0; i < start[0].Length; i++)
        {
            double full = afterReference[0][i] - start[0][i];
            double half = afterScaled[0][i] - start[0][i];
            moved += Math.Abs(full);
            Assert.Equal(0.5 * full, half, 12);
        }
        Assert.True(moved > 0, "the reference step did not move the scaled layer at all");
        Assert.Equal(afterReference[1], afterScaled[1]); // the other layer takes the base-rate step either way
    }

    [Fact]
    public void A_cap_limits_the_rate_and_invalid_policies_are_refused()
    {
        var layer = new DenseLayer<double>(2, (IActivationFunction<double>)new IdentityActivation<double>());
        Assert.Throws<ArgumentOutOfRangeException>(() => layer.LearningRateScale = 0);
        Assert.Throws<ArgumentOutOfRangeException>(() => layer.MaxLearningRate = -1e-3);
        layer.MaxLearningRate = 1e-3;
        Assert.Equal(1e-3, layer.MaxLearningRate);
    }
}
