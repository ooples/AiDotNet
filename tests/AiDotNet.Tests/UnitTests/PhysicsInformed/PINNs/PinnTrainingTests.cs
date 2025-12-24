using System;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.PhysicsInformed.Interfaces;
using AiDotNet.PhysicsInformed.PINNs;
using Xunit;

namespace AiDotNet.Tests.UnitTests.PhysicsInformed.PINNs;

public class PinnTrainingTests
{
    [Fact]
    public void PhysicsInformedNeuralNetwork_SolveUpdatesParameters()
    {
        var architecture = CreateLinearArchitecture(inputSize: 1, outputSize: 1);
        var pde = new LinearResidualPde();
        var model = new PhysicsInformedNeuralNetwork<double>(
            architecture,
            pde,
            Array.Empty<IBoundaryCondition<double>>(),
            initialCondition: null,
            numCollocationPoints: 8);

        var before = model.GetParameters().ToArray();
        var history = model.Solve(epochs: 1, learningRate: 0.01, verbose: false, batchSize: 4);
        var after = model.GetParameters().ToArray();

        Assert.Single(history.Losses);
        Assert.False(before.SequenceEqual(after));
    }

    [Fact]
    public void DeepRitzMethod_SolveUpdatesParameters()
    {
        var architecture = CreateLinearArchitecture(inputSize: 1, outputSize: 1);
        var model = new DeepRitzMethod<double>(
            architecture,
            EnergyFunctional,
            boundaryCheck: null,
            boundaryValue: null,
            numQuadraturePoints: 8);

        var before = model.GetParameters().ToArray();
        var history = model.Solve(epochs: 1, learningRate: 0.01, verbose: false, batchSize: 4, derivativeStep: 1e-4);
        var after = model.GetParameters().ToArray();

        Assert.Single(history.Losses);
        Assert.False(before.SequenceEqual(after));
    }

    [Fact]
    public void VariationalPinn_SolveUpdatesParameters()
    {
        var architecture = CreateLinearArchitecture(inputSize: 1, outputSize: 1);
        var model = new VariationalPINN<double>(
            architecture,
            WeakFormResidual,
            numQuadraturePoints: 8,
            numTestFunctions: 2);

        var before = model.GetParameters().ToArray();
        var history = model.Solve(epochs: 1, learningRate: 0.01, verbose: false, batchSize: 4, derivativeStep: 1e-4);
        var after = model.GetParameters().ToArray();

        Assert.Single(history.Losses);
        Assert.False(before.SequenceEqual(after));
    }

    private static double EnergyFunctional(double[] x, double[] u, double[,] gradU)
    {
        double diff = u[0] - x[0];
        return diff * diff;
    }

    private static double WeakFormResidual(double[] x, double[] u, double[,] gradU, double[] v, double[,] gradV)
    {
        double diff = u[0] - x[0];
        return diff * v[0];
    }

    private static NeuralNetworkArchitecture<double> CreateLinearArchitecture(int inputSize, int outputSize)
    {
        var layers = new ILayer<double>[]
        {
            new DenseLayer<double>(inputSize, outputSize, (IActivationFunction<double>)new IdentityActivation<double>())
        };

        return new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: inputSize,
            outputSize: outputSize,
            layers: layers.ToList());
    }

    private sealed class LinearResidualPde : IPDESpecification<double>, IPDEResidualGradient<double>
    {
        public double ComputeResidual(double[] inputs, double[] outputs, PDEDerivatives<double> derivatives)
        {
            return outputs[0] - inputs[0];
        }

        public int InputDimension => 1;
        public int OutputDimension => 1;
        public string Name => "LinearResidualTest";

        public PDEResidualGradient<double> ComputeResidualGradient(
            double[] inputs,
            double[] outputs,
            PDEDerivatives<double> derivatives)
        {
            var gradient = new PDEResidualGradient<double>(OutputDimension, InputDimension);
            gradient.OutputGradients[0] = 1.0;
            return gradient;
        }
    }
}
