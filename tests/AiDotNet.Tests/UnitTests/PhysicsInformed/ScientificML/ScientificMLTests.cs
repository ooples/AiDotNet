using AiDotNet.Tensors.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.PhysicsInformed.Options;
using AiDotNet.PhysicsInformed.ScientificML;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.UnitTests.PhysicsInformed.ScientificML;

public class ScientificMLTests
{
    [Fact(Timeout = 60000)]
    public async Task HamiltonianNeuralNetwork_ComputeTimeDerivative_UsesHamiltonianGradient()
    {
        var layer = CreateLinearLayer(inputSize: 2, outputSize: 1);
        layer.SetParameters(new Vector<double>(new[] { 1.0, 1.0, 0.0 }));

        var architecture = CreateArchitecture(2, 1, new List<ILayer<double>> { layer });
        var model = new HamiltonianNeuralNetwork<double>(architecture, stateDim: 2);

        var derivative = model.ComputeTimeDerivative(new Vector<double>(new[] { 0.2, -0.4 }));

        Assert.Equal(2, derivative.Length);
        Assert.InRange(derivative[0], 0.999, 1.001);
        Assert.InRange(derivative[1], -1.001, -0.999);
    }

    [Fact(Timeout = 60000)]
    public async Task HamiltonianNeuralNetwork_TrainUpdatesParameters()
    {
        var layer = CreateLinearLayer(inputSize: 2, outputSize: 1);
        layer.SetParameters(new Vector<double>(new[] { 0.4, -0.2, 0.1 }));

        var architecture = CreateArchitecture(2, 1, new List<ILayer<double>> { layer });
        var model = new HamiltonianNeuralNetwork<double>(architecture, stateDim: 2);

        var input = new Tensor<double>(new[] { 1, 2 });
        input[0, 0] = 0.25;
        input[0, 1] = -0.5;

        // The HNN paper trains the symplectic time derivative, not scalar H labels.
        // For H(q,p) = 0.5(q^2+p^2), the canonical target is (p,-q).
        var target = new Tensor<double>(new[] { 1, 2 });
        target[0, 0] = input[0, 1];
        target[0, 1] = -input[0, 0];

        var before = model.GetParameters().ToArray();
        model.Train(input, target);
        var after = model.GetParameters().ToArray();

        Assert.False(before.SequenceEqual(after));
    }

    [Fact]
    public void HamiltonianNeuralNetwork_TrainingObjective_IsTheSymplecticDerivative()
    {
        var layer = CreateLinearLayer(inputSize: 2, outputSize: 1);
        layer.SetParameters(new Vector<double>(new[] { 1.0, 1.0, 0.0 }));

        var architecture = CreateArchitecture(2, 1, new List<ILayer<double>> { layer });
        var model = new HamiltonianNeuralNetwork<double>(architecture, stateDim: 2);
        var objective = Assert.IsAssignableFrom<ITrainingObjectiveProvider<double>>(model);
        var input = new Tensor<double>(new[] { 1, 2 });
        input[0, 0] = 0.2;
        input[0, 1] = -0.4;
        var target = new Tensor<double>(new[] { 1, 2 });
        target[0, 0] = 1.0;
        target[0, 1] = -1.0;

        Assert.Equal(TrainingObjectiveKind.HamiltonianDynamics, objective.TrainingObjectiveKind);
        Assert.Same(target, objective.ResolveTrainingTarget(input, target));
        Assert.InRange(objective.EvaluateTrainingObjective(input, target), 0.0, 1e-18);
        Assert.Throws<ArgumentException>(() =>
            objective.ResolveTrainingTarget(input, new Tensor<double>(new[] { 1, 1 })));
    }

    [Fact]
    public void HamiltonianNeuralNetwork_DefaultArchitecture_MatchesReferenceCoordinateMlp()
    {
        var options = new HamiltonianNeuralNetworkOptions();
        var model = new HamiltonianNeuralNetwork<double>();

        Assert.Equal(2, options.HiddenLayerCount);
        Assert.Equal(200, options.HiddenDimension);
        Assert.Equal(1e-3, options.LearningRate, precision: 12);
        Assert.Equal(1e-4, options.WeightDecay, precision: 12);
        Assert.Equal(41_000, model.GetParameterCount());
    }

    [Fact(Timeout = 60000)]
    public async Task LagrangianNeuralNetwork_ComputeAcceleration_ConstantLagrangianReturnsZero()
    {
        // L(q, qDot) = -tanh(qDot + 0.5) + 1 has dL/dq = 0, d^2L/(dq dqDot) = 0, and positive mass
        // d^2L/dqDot^2 = 2 * tanh(qDot + 0.5) * sech^2(qDot + 0.5) > 0 at qDot = -0.2, so qDDot = 0.
        var kineticHidden = new DenseLayer<double>(1, (IActivationFunction<double>)new TanhActivation<double>());
        kineticHidden.Forward(new Tensor<double>(new[] { 1, 2 }));
        kineticHidden.SetParameters(new Vector<double>(new[] { 0.0, 1.0, 0.5 }));
        var readout = CreateLinearLayer(inputSize: 1, outputSize: 1);
        readout.Forward(new Tensor<double>(new[] { 1, 1 }));
        readout.SetParameters(new Vector<double>(new[] { -1.0, 1.0 }));

        var architecture = CreateArchitecture(2, 1, new List<ILayer<double>> { kineticHidden, readout });
        var model = new LagrangianNeuralNetwork<double>(architecture, configurationDim: 1);

        var q = new Vector<double>(new[] { 0.5 });
        var qDot = new Vector<double>(new[] { -0.2 });
        var acceleration = model.ComputeAcceleration(q, qDot);

        Assert.Single(acceleration);
        Assert.InRange(Math.Abs(acceleration[0]), 0.0, 1e-6);

        // A genuinely constant L(q, qDot) = 1 has identically zero Hessian mass matrix (0 / 0) and must
        // surface a non-finite acceleration rather than silently regularizing to zero.
        var singularLayer = CreateLinearLayer(inputSize: 2, outputSize: 1);
        singularLayer.SetParameters(new Vector<double>(new[] { 0.0, 0.0, 1.0 }));
        var singularModel = new LagrangianNeuralNetwork<double>(
            CreateArchitecture(2, 1, new List<ILayer<double>> { singularLayer }),
            configurationDim: 1);
        var singularAcceleration = singularModel.ComputeAcceleration(q, qDot);
        Assert.Single(singularAcceleration);
        Assert.True(double.IsNaN(singularAcceleration[0]) || double.IsInfinity(singularAcceleration[0]));
    }

    [Fact(Timeout = 60000)]
    public async Task LagrangianNeuralNetwork_TrainUpdatesParameters()
    {
        var hidden = new DenseLayer<double>(2, (IActivationFunction<double>)new TanhActivation<double>());
        hidden.Forward(new Tensor<double>(new[] { 1, 2 }));
        hidden.SetParameters(new Vector<double>(new[] { 0.3, -0.2, 0.8, 0.5, 0.4, -0.3 }));
        var output = CreateLinearLayer(inputSize: 2, outputSize: 1);
        output.Forward(new Tensor<double>(new[] { 1, 2 }));
        output.SetParameters(new Vector<double>(new[] { 0.6, -0.4, 0.2 }));

        var architecture = CreateArchitecture(2, 1, new List<ILayer<double>> { hidden, output });
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

    [Fact(Timeout = 60000)]
    public async Task UniversalDifferentialEquation_RungeKutta4ImprovesAccuracy()
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

    [Fact(Timeout = 60000)]
    public async Task UniversalDifferentialEquation_TrainUpdatesParameters()
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

    [Fact(Timeout = 60000)]
    public async Task SymbolicPhysicsLearner_DiscoversSimpleLinearRelationship()
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
