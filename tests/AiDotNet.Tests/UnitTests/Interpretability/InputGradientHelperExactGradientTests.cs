using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Interpretability.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.Interpretability;

/// <summary>
/// Pins the exact-gradient paths of <see cref="InputGradientHelper{T}"/> after the legacy autodiff removal: tape gradients
/// for an engine-op tensor function (and when they must be rejected), and tensor gradients for a neural network.
/// </summary>
public class InputGradientHelperExactGradientTests
{
    private static readonly Tensor<double> Weights = new(new[] { 0.5, -1.0, 2.0, 0.25 }, new[] { 4, 1 });

    private static Tensor<double> TanhOfLinear(Tensor<double> x) =>
        AiDotNetEngine.Current.Tanh(AiDotNetEngine.Current.TensorMatMul(x, Weights));

    private static double[] ExpectedTanhOfLinearGradient(double[] x)
    {
        double z = 0;
        for (int i = 0; i < x.Length; i++) z += x[i] * Weights[i, 0];
        double dt = 1 - Math.Tanh(z) * Math.Tanh(z);
        var g = new double[x.Length];
        for (int i = 0; i < x.Length; i++) g[i] = Weights[i, 0] * dt;
        return g;
    }

    [Fact]
    public void OutOfRangeOutputIndex_ReturnsZeros_AndDoesNotDisableExactGradients()
    {
        Func<Tensor<double>, Tensor<double>> f = TanhOfLinear;
        var helper = new InputGradientHelper<double>(f);
        var x = new[] { 0.1, 0.2, -0.3, 0.4 };
        var x0 = new Tensor<double>(x, new[] { 1, 4 });

        // The function has one output: index 3 has no gradient. That must not be mistaken for "not on the tape".
        var none = helper.ComputeGradientTensor(x0, 3);
        Assert.All(none.ToArray(), v => Assert.Equal(0.0, v));

        var g = helper.ComputeGradientTensor(x0, 0);
        Assert.True(helper.SupportsExactGradients);
        Assert.Equal("Backpropagation", helper.GradientMethod);
        var expected = ExpectedTanhOfLinearGradient(x);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], g[i], 12);
    }

    [Fact]
    public void TapeConnectedFunction_ThatReadsItsInputOnTheHost_FailsVerification_AndUsesNumericalGradients()
    {
        // output = x * x[0] with x[0] read on the host: the tape sees a constant scale and reports d/dx0 = x0, but the
        // function is x0^2 in its first output, so the true derivative is 2 * x0. The directional check must catch it.
        Func<Tensor<double>, Tensor<double>> f = x => AiDotNetEngine.Current.TensorMultiplyScalar(x, x.ToArray()[0]);
        var helper = new InputGradientHelper<double>(f);

        var g = helper.ComputeGradientTensor(new Tensor<double>(new[] { 10.0, 0.5 }, new[] { 1, 2 }), 0);

        Assert.False(helper.SupportsExactGradients);
        Assert.Equal("NumericalApproximation", helper.GradientMethod);
        Assert.Equal(20.0, g[0], 4);
        Assert.Equal(0.0, g[1], 4);
    }

    [Fact]
    public void NeuralNetwork_ComputeGradientTensor_MatchesTheVectorGradient()
    {
        var net = new FeedForwardNeuralNetwork<double>(new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputSize: 3, outputSize: 1, layers: new List<ILayer<double>>
            {
                new InputLayer<double>(3),
                new DenseLayer<double>(4, activationFunction: new TanhActivation<double>()),
                new DenseLayer<double>(1, activationFunction: new IdentityActivation<double>()),
            }), lossFunction: new MeanSquaredErrorLoss<double>());
        var helper = new InputGradientHelper<double>(net);
        var values = new[] { 0.2, -0.5, 0.9 };

        var fromVector = helper.ComputeGradient(new Vector<double>(values), 0);
        var fromTensor = helper.ComputeGradientTensor(new Tensor<double>(values, new[] { 1, 3 }), 0);

        Assert.Equal(new[] { 1, 3 }, fromTensor.Shape.ToArray());
        for (int i = 0; i < values.Length; i++) Assert.Equal(fromVector[i], fromTensor[i], 12);
        Assert.Contains(fromTensor.ToArray(), v => v != 0);
    }
}
