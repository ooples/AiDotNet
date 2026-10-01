using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>A copy owns its optimizer: it must not train with the original's optimizer or its moment state.</summary>
public class DeepCopyOptimizerOwnershipTests
{
    private static NeuralNetwork<double> Network() => new(new NeuralNetworkArchitecture<double>(
        InputType.OneDimensional, NeuralNetworkTaskType.Regression, NetworkComplexity.Simple,
        inputSize: 4, outputSize: 2, layers: new List<ILayer<double>>
        {
            new DenseLayer<double>(6, (IActivationFunction<double>)new TanhActivation<double>()),
            new DenseLayer<double>(2, (IActivationFunction<double>)new IdentityActivation<double>()),
        }));

    private static object? Optimizer(NeuralNetwork<double> n) =>
        typeof(NeuralNetwork<double>).GetField("_optimizer", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)?.GetValue(n);

    [Fact]
    public void A_copy_trains_with_its_own_fresh_optimizer()
    {
        var x = new Tensor<double>(new[] { 3, 4 });
        var y = new Tensor<double>(new[] { 3, 2 });
        for (int i = 0; i < x.Length; i++) x[i] = Math.Sin(i + 1.0);
        for (int i = 0; i < y.Length; i++) y[i] = Math.Cos(i + 1.0);

        var original = Network();
        original.Predict(x);
        original.Train(x, y); // the original's optimizer now carries moment state
        var copy = (NeuralNetwork<double>)original.DeepCopy();

        var copyOptimizer = Optimizer(copy);
        Assert.NotNull(copyOptimizer);
        Assert.NotSame(Optimizer(original), copyOptimizer);
        Assert.Same(copy, (copyOptimizer as AiDotNet.Optimizers.OptimizerBase<double, Tensor<double>, Tensor<double>>)?.Model);

        // Fresh state: the copy's first step equals a fresh model's first step from the same weights.
        var fresh = Network();
        fresh.Predict(x);
        fresh.SetParameters(copy.GetParameters());
        var before = copy.GetParameters().ToArray();
        copy.Train(x, y);
        fresh.Train(x, y);
        Assert.Equal(fresh.GetParameters().ToArray(), copy.GetParameters().ToArray());
        Assert.NotEqual(before, copy.GetParameters().ToArray());
    }
}
