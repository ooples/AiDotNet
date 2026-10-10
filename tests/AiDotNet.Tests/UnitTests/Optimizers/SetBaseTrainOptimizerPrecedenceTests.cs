using System;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// An explicit SetBaseTrainOptimizer must win over the optimizer a model's Train override passes to TrainWithTape
/// (its constructor default). FeedForwardNeuralNetwork defaults to AMSGrad-Adam and calls
/// TrainWithTape(input, expected, _optimizer), as ~616 models do; before the fix the configured optimizer was
/// silently ignored on all of them.
/// </summary>
[Collection("NonParallelIntegration")]
public sealed class SetBaseTrainOptimizerPrecedenceTests
{
    [Fact]
    public void ExplicitBaseOptimizer_OverridesTheModelsConstructorDefault()
    {
        const double LearningRate = 0.05;
        var configured = Build(optimizer: null);
        configured.SetBaseTrainOptimizer(Sgd(LearningRate));
        var reference = Build(optimizer: Sgd(LearningRate));
        reference.SetParameters(configured.GetParameters());
        var start = configured.GetParameters();

        var (input, target) = Data();
        configured.Train(input, target);
        reference.Train(input, target);

        var trained = configured.GetParameters();
        var expected = reference.GetParameters();
        Assert.Equal(expected.Length, trained.Length);
        bool moved = false;
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(Math.Abs(expected[i] - trained[i]) <= 1e-6f,
                $"parameter {i}: configured SGD model {trained[i]} vs reference SGD model {expected[i]}; " +
                "the explicitly configured optimizer was not the one that trained");
            moved |= trained[i] != start[i];
        }
        Assert.True(moved, "no parameter changed, so this test would prove nothing");
    }

    private static StochasticGradientDescentOptimizer<float, Tensor<float>, Tensor<float>> Sgd(double learningRate) =>
        new(null, new StochasticGradientDescentOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            InitialLearningRate = learningRate,
        });

    private static FeedForwardNeuralNetwork<float> Build(
        IGradientBasedOptimizer<float, Tensor<float>, Tensor<float>>? optimizer)
    {
        var layers = new System.Collections.Generic.List<ILayer<float>>
        {
            new DenseLayer<float>(12, activationFunction: new ReLUActivation<float>()),
            new DenseLayer<float>(3, activationFunction: (IActivationFunction<float>?)null),
        };
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: 6,
            outputSize: 3,
            layers: layers);
        var model = new FeedForwardNeuralNetwork<float>(architecture, optimizer,
            lossFunction: new CrossEntropyWithLogitsLoss<float>());
        model.SetTrainingMode(true);
        return model;
    }

    private static (Tensor<float> Input, Tensor<float> Target) Data()
    {
        var rng = new Random(17);
        var input = new Tensor<float>(new[] { 8, 6 });
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
        var target = new Tensor<float>(new[] { 8, 3 });
        for (int row = 0; row < 8; row++) target[row * 3 + row % 3] = 1f;
        return (input, target);
    }
}
