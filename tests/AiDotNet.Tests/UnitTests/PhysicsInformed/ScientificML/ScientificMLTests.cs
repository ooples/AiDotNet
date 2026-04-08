using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.PhysicsInformed.ScientificML;
using Xunit;

namespace AiDotNet.Tests.UnitTests.PhysicsInformed.ScientificML;

public class ScientificMLTests
{
    [Fact]
    public void HamiltonianNeuralNetwork_ComputeTimeDerivative_UsesHamiltonianGradient()
    {
        var layer = CreateLinearLayer(inputSize: 2, outputSize: 1);
        layer.SetParameters(new Vector<double>(new[] { 1.0, 1.0, 0.0 }));

        var architecture = CreateArchitecture(2, 1, new List<ILayer<double>> { layer });
        var model = new HamiltonianNeuralNetwork<double>(architecture, stateDim: 2);

        var derivative = model.ComputeTimeDerivative(new[] { 0.2, -0.4 });

        Assert.Equal(2, derivative.Length);
        Assert.InRange(derivative[0], 0.999, 1.001);
        Assert.InRange(derivative[1], -1.001, -0.999);
    }

    [Fact]
    public void HamiltonianNeuralNetwork_TrainUpdatesParameters()
    {
        var layer = CreateLinearLayer(inputSize: 2, outputSize: 1);
        layer.SetParameters(new Vector<double>(new[] { 0.4, -0.2, 0.1 }));

        var architecture = CreateArchitecture(2, 1, new List<ILayer<double>> { layer });
        var model = new HamiltonianNeuralNetwork<double>(architecture, stateDim: 2);

        var input = new Tensor<double>(new[] { 1, 2 });
        input[0, 0] = 0.25;
        input[0, 1] = -0.5;

        var target = new Tensor<double>(new[] { 1, 1 });
        target[0, 0] = 0.0;

        var before = model.GetParameters().ToArray();
        model.Train(input, target);
        var after = model.GetParameters().ToArray();

        Assert.False(before.SequenceEqual(after));
    }

    [Fact]
    public void LagrangianNeuralNetwork_ComputeAcceleration_ConstantLagrangianReturnsZero()
    {
        var layer = CreateLinearLayer(inputSize: 2, outputSize: 1);
        layer.SetParameters(new Vector<double>(new[] { 0.0, 0.0, 1.0 }));

        var architecture = CreateArchitecture(2, 1, new List<ILayer<double>> { layer });
        var model = new LagrangianNeuralNetwork<double>(architecture, configurationDim: 1);

        var acceleration = model.ComputeAcceleration(new[] { 0.5 }, new[] { -0.2 });

        Assert.Single(acceleration);
        Assert.InRange(Math.Abs(acceleration[0]), 0.0, 1e-6);
    }

    [Fact]
    public void LagrangianNeuralNetwork_TrainUpdatesParameters()
    {
        var layer = CreateLinearLayer(inputSize: 2, outputSize: 1);
        layer.SetParameters(new Vector<double>(new[] { 0.3, -0.4, 0.2 }));

        var architecture = CreateArchitecture(2, 1, new List<ILayer<double>> { layer });
        var model = new LagrangianNeuralNetwork<double>(architecture, configurationDim: 1);

        var input = new Tensor<double>(new[] { 1, 2 });
        input[0, 0] = 0.15;
        input[0, 1] = -0.25;

        var target = new Tensor<double>(new[] { 1, 1 });
        target[0, 0] = 0.1;

        var before = model.GetParameters().ToArray();
        model.Train(input, target);
        var after = model.GetParameters().ToArray();

        Assert.False(before.SequenceEqual(after));
    }

    [Fact]
    public void UniversalDifferentialEquation_RungeKutta4ImprovesAccuracy()
    {
        var layer = CreateLinearLayer(inputSize: 2, outputSize: 1);
        layer.SetParameters(new Vector<double>(new[] { 0.0, 0.0, 0.0 }));

        var architecture = CreateArchitecture(2, 1, new List<ILayer<double>> { layer });
        var model = new UniversalDifferentialEquation<double>(
            architecture,
            stateDim: 1,
            knownDynamics: (state, time) => new[] { -state[0] });

        var euler = model.Simulate(new[] { 1.0 }, 0.0, 1.0, 1, OdeIntegrationMethod.Euler);
        var rk4 = model.Simulate(new[] { 1.0 }, 0.0, 1.0, 1, OdeIntegrationMethod.RungeKutta4);

        double expected = Math.Exp(-1.0);
        double eulerError = Math.Abs(euler[1, 0] - expected);
        double rk4Error = Math.Abs(rk4[1, 0] - expected);

        Assert.True(rk4Error < eulerError);
    }

    [Fact]
    public void UniversalDifferentialEquation_TrainUpdatesParameters()
    {
        var layer = CreateLinearLayer(inputSize: 2, outputSize: 1);
        layer.SetParameters(new Vector<double>(new[] { 0.15, -0.05, 0.02 }));

        var architecture = CreateArchitecture(2, 1, new List<ILayer<double>> { layer });
        var model = new UniversalDifferentialEquation<double>(
            architecture,
            stateDim: 1,
            knownDynamics: (state, time) => new[] { 0.0 });

        var input = new Tensor<double>(new[] { 1, 2 });
        input[0, 0] = 0.2;
        input[0, 1] = 0.1;

        var target = new Tensor<double>(new[] { 1, 1 });
        target[0, 0] = 0.3;

        var before = model.GetParameters().ToArray();
        model.Train(input, target);
        var after = model.GetParameters().ToArray();

        Assert.False(before.SequenceEqual(after));
    }

    [Fact]
    public void SymbolicPhysicsLearner_DiscoversSimpleLinearRelationship()
    {
        var learner = new SymbolicPhysicsLearner<double>();
        var inputs = new double[,]
        {
            { 0.0 },
            { 1.0 },
            { 2.0 }
        };
        var outputs = new[] { 0.0, 2.0, 4.0 };

        var expression = learner.DiscoverEquation(inputs, outputs, maxComplexity: 4, numGenerations: 25);

        double prediction1 = expression.Evaluate(new Dictionary<string, double> { { "x1", 1.0 } });
        double prediction2 = expression.Evaluate(new Dictionary<string, double> { { "x1", 2.0 } });

        Assert.InRange(prediction1, 1.999, 2.001);
        Assert.InRange(prediction2, 3.999, 4.001);
    }

    private static DenseLayer<double> CreateLinearLayer(int inputSize, int outputSize)
    {
        return new DenseLayer<double>(
            inputSize,
            outputSize,
            (IActivationFunction<double>)new IdentityActivation<double>());
    }

    private static NeuralNetworkArchitecture<double> CreateArchitecture(int inputSize, int outputSize, List<ILayer<double>> layers)
    {
        return new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: inputSize,
            outputSize: outputSize,
            layers: layers);
    }
}
