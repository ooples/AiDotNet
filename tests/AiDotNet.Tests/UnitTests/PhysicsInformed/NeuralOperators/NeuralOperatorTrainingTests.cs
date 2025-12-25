using System.Collections.Generic;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.PhysicsInformed.NeuralOperators;
using Xunit;

namespace AiDotNet.Tests.UnitTests.PhysicsInformed.NeuralOperators;

public class NeuralOperatorTrainingTests
{
    [Fact]
    public void FourierNeuralOperator_TrainUpdatesParameters()
    {
        var architecture = CreateLinearArchitecture(inputSize: 2, outputSize: 2);
        var model = new FourierNeuralOperator<double>(
            architecture,
            modes: 1,
            width: 2,
            spatialDimensions: new[] { 2 },
            numLayers: 1);

        var input = new Tensor<double>(new[] { 1, 2, 2 });
        input[0, 0, 0] = 0.1;
        input[0, 0, 1] = 0.2;
        input[0, 1, 0] = 0.3;
        input[0, 1, 1] = 0.4;

        var target = new Tensor<double>(new[] { 1, 2, 2 });
        target[0, 0, 0] = 0.2;
        target[0, 0, 1] = 0.4;
        target[0, 1, 0] = 0.6;
        target[0, 1, 1] = 0.8;

        var before = model.GetParameters().ToArray();
        var history = model.Train(new[] { input }, new[] { target }, epochs: 1, learningRate: 0.01);
        var after = model.GetParameters().ToArray();

        Assert.Single(history.Losses);
        Assert.False(before.SequenceEqual(after));
    }

    [Fact]
    public void DeepOperatorNetwork_TrainUpdatesParameters()
    {
        var branchArchitecture = CreateLinearArchitecture(inputSize: 2, outputSize: 2);
        var trunkArchitecture = CreateLinearArchitecture(inputSize: 1, outputSize: 2);
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 3,
            outputSize: 1);

        var model = new DeepOperatorNetwork<double>(
            architecture,
            branchArchitecture,
            trunkArchitecture,
            latentDimension: 2,
            numSensors: 2);

        var inputFunctions = new double[,] { { 0.1, 0.2 } };
        var queryLocations = new double[1, 1, 1];
        queryLocations[0, 0, 0] = 0.3;
        var targetValues = new double[,] { { 0.5 } };

        var before = model.GetParameters().ToArray();
        var history = model.Train(inputFunctions, queryLocations, targetValues, epochs: 1, learningRate: 0.01, verbose: false);
        var after = model.GetParameters().ToArray();

        Assert.Single(history.Losses);
        Assert.False(before.SequenceEqual(after));
    }

    [Fact]
    public void GraphNeuralOperator_TrainOnGraphUpdatesParameters()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 2,
            outputSize: 2);

        var model = new GraphNeuralOperator<double>(architecture, numLayers: 1, hiddenDim: 2);

        var nodeFeatures = new double[,]
        {
            { 0.1, 0.2 },
            { 0.0, -0.1 }
        };
        var adjacency = new double[,]
        {
            { 1.0, 0.5 },
            { 0.5, 1.0 }
        };
        var targets = new double[,]
        {
            { 0.2, 0.1 },
            { 0.0, 0.0 }
        };

        var before = model.GetParameters().ToArray();
        var history = model.TrainOnGraph(nodeFeatures, adjacency, targets, epochs: 1, learningRate: 0.01, verbose: false);
        var after = model.GetParameters().ToArray();

        Assert.Single(history.Losses);
        Assert.False(before.SequenceEqual(after));
    }

    private static NeuralNetworkArchitecture<double> CreateLinearArchitecture(int inputSize, int outputSize)
    {
        var layers = new List<ILayer<double>>
        {
            new DenseLayer<double>(
                inputSize,
                outputSize,
                (IActivationFunction<double>)new IdentityActivation<double>())
        };

        return new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: inputSize,
            outputSize: outputSize,
            layers: layers);
    }
}
