using System;
using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.PhysicsInformed;

public class NeuralNetworkDerivativesTests
{
    [Fact]
    public void ComputeDerivatives_LinearLayer_ReturnsConstantGradientAndZeroHessian()
    {
        var layer = new DenseLayer<double>(2, 1, (IActivationFunction<double>)new IdentityActivation<double>());
        layer.SetParameters(new Vector<double>(new[] { 2.0, 3.0, 1.0 }));

        var network = CreateNetwork(new List<ILayer<double>> { layer }, inputSize: 2, outputSize: 1);
        var derivatives = NeuralNetworkDerivatives<double>.ComputeDerivatives(network, new[] { 0.4, -0.2 }, 1);
        var first = derivatives.FirstDerivatives!;
        var second = derivatives.SecondDerivatives!;

        Assert.InRange(first[0, 0], 1.999999, 2.000001);
        Assert.InRange(first[0, 1], 2.999999, 3.000001);
        Assert.InRange(Math.Abs(second[0, 0, 0]), 0.0, 1e-9);
        Assert.InRange(Math.Abs(second[0, 0, 1]), 0.0, 1e-9);
        Assert.InRange(Math.Abs(second[0, 1, 0]), 0.0, 1e-9);
        Assert.InRange(Math.Abs(second[0, 1, 1]), 0.0, 1e-9);
    }

    [Fact]
    public void ComputeGradient_LinearLayer_ReturnsExpectedJacobian()
    {
        var layer = new DenseLayer<double>(2, 2, (IActivationFunction<double>)new IdentityActivation<double>());
        layer.SetParameters(new Vector<double>(new[] { 1.0, 2.0, 0.5, -1.0, -2.0, 0.25 }));

        var network = CreateNetwork(new List<ILayer<double>> { layer }, inputSize: 2, outputSize: 2);
        var gradient = NeuralNetworkDerivatives<double>.ComputeGradient(network, new[] { 0.1, -0.3 }, 2);

        Assert.InRange(gradient[0, 0], 0.999999, 1.000001);
        Assert.InRange(gradient[0, 1], 1.999999, 2.000001);
        Assert.InRange(gradient[1, 0], 0.499999, 0.500001);
        Assert.InRange(gradient[1, 1], -1.000001, -0.999999);
    }

    [Fact]
    public void ComputeDerivatives_TanhLayer_ReturnsExpectedGradientAndHessian()
    {
        var layer = new DenseLayer<double>(2, 1, (IActivationFunction<double>)new TanhActivation<double>());
        layer.SetParameters(new Vector<double>(new[] { 0.5, -1.0, 0.1 }));

        var network = CreateNetwork(new List<ILayer<double>> { layer }, inputSize: 2, outputSize: 1);
        var input = new[] { 0.2, -0.4 };
        var derivatives = NeuralNetworkDerivatives<double>.ComputeDerivatives(network, input, 1);
        var tanhFirst = derivatives.FirstDerivatives!;
        var tanhSecond = derivatives.SecondDerivatives!;

        double z = 0.5 * input[0] + (-1.0) * input[1] + 0.1;
        double t = Math.Tanh(z);
        double first = 1.0 - t * t;
        double second = -2.0 * t * first;

        double expectedG0 = first * 0.5;
        double expectedG1 = first * -1.0;
        double expectedH00 = second * 0.5 * 0.5;
        double expectedH01 = second * 0.5 * -1.0;
        double expectedH11 = second * -1.0 * -1.0;

        Assert.InRange(tanhFirst[0, 0], expectedG0 - 1e-6, expectedG0 + 1e-6);
        Assert.InRange(tanhFirst[0, 1], expectedG1 - 1e-6, expectedG1 + 1e-6);
        Assert.InRange(tanhSecond[0, 0, 0], expectedH00 - 1e-6, expectedH00 + 1e-6);
        Assert.InRange(tanhSecond[0, 0, 1], expectedH01 - 1e-6, expectedH01 + 1e-6);
        Assert.InRange(tanhSecond[0, 1, 0], expectedH01 - 1e-6, expectedH01 + 1e-6);
        Assert.InRange(tanhSecond[0, 1, 1], expectedH11 - 1e-6, expectedH11 + 1e-6);
    }

    private static NeuralNetwork<double> CreateNetwork(List<ILayer<double>> layers, int inputSize, int outputSize)
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: inputSize,
            outputSize: outputSize,
            layers: layers);

        return new NeuralNetwork<double>(architecture);
    }
}

