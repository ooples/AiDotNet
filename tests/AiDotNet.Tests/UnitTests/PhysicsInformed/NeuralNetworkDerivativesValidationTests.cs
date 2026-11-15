using System;
using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.PhysicsInformed;
using Xunit;

namespace AiDotNetTests.UnitTests.PhysicsInformed;

/// <summary>Argument validation of <see cref="NeuralNetworkDerivatives{T}"/>.</summary>
public class NeuralNetworkDerivativesValidationTests
{
    private static FeedForwardNeuralNetwork<double> SingleOutputNetwork() => new(new NeuralNetworkArchitecture<double>(
        inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
        inputSize: 2, outputSize: 1, layers: new List<ILayer<double>>
        {
            new InputLayer<double>(2),
            new DenseLayer<double>(3, activationFunction: new TanhActivation<double>()),
            new DenseLayer<double>(1, activationFunction: new IdentityActivation<double>()),
        }), lossFunction: new MeanSquaredErrorLoss<double>());

    [Fact]
    public void ComputeDerivatives_OutputDimBeyondTheNetwork_IsRejectedOnOutputDim()
    {
        var error = Assert.Throws<ArgumentOutOfRangeException>(
            () => NeuralNetworkDerivatives<double>.ComputeDerivatives(SingleOutputNetwork(), new[] { 0.3, -0.7 }, 2));
        Assert.Equal("outputDim", error.ParamName);
    }

    [Fact]
    public void ComputeDerivatives_AllOutputs_HasTheDocumentedShapes()
    {
        var d = NeuralNetworkDerivatives<double>.ComputeDerivatives(SingleOutputNetwork(), new[] { 0.3, -0.7 }, 1);
        var first = d.FirstDerivatives ?? throw new InvalidOperationException("FirstDerivatives is null.");
        var second = d.SecondDerivatives ?? throw new InvalidOperationException("SecondDerivatives is null.");
        Assert.Equal(new[] { 1, 2 }, first.Shape.ToArray());
        Assert.Equal(new[] { 1, 2, 2 }, second.Shape.ToArray());
        // A Hessian of a smooth scalar function is symmetric.
        Assert.Equal(second[0, 0, 1], second[0, 1, 0], 10);
    }
}
