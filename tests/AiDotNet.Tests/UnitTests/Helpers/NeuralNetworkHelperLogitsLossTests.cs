using System;
using System.Collections.Generic;
using AiDotNet.Enums;
using AiDotNet.Configuration;
using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Engines.Autodiff;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Helpers;

/// <summary>
/// The default loss for a raw-logits head is the with-logits form of the probability default, for every task type, and a
/// network keeps the probability default for a head the caller supplied.
/// </summary>
public class NeuralNetworkHelperLogitsLossTests
{
    [Fact]
    public void ResNetGeneratedHeadReturnsLogitsForItsDefaultLoss()
    {
        var architecture = new NeuralNetworkArchitecture<double>(InputType.ThreeDimensional,
            NeuralNetworkTaskType.ImageClassification, inputHeight: 32, inputWidth: 32, inputDepth: 3,
            outputSize: 3);
        var configuration = new ResNetConfiguration(ResNetVariant.ResNet18, numClasses: 3,
            inputHeight: 32, inputWidth: 32);
        using var network = new ResNetNetwork<double>(architecture, configuration);
        Assert.IsType<CrossEntropyWithLogitsLoss<double>>(network.DefaultLossFunction);

        var head = network.Classifier!;
        var parameters = new Vector<double>(head.ParameterCount);
        double[] logits = [-2.0, 0.25, 1.5];
        for (int i = 0; i < logits.Length; i++)
            parameters[parameters.Length - logits.Length + i] = logits[i];
        head.UpdateParameters(parameters);
        var output = head.Forward(new Tensor<double>([1, configuration.Trunk.OutputChannels[3]]));
        for (int i = 0; i < logits.Length; i++)
            Assert.Equal(logits[i], output.GetFlat(i), 12);
    }

    [Fact]
    public void EveryTaskGetsTheWithLogitsFormOfItsProbabilityDefault()
    {
        foreach (NeuralNetworkTaskType task in Enum.GetValues(typeof(NeuralNetworkTaskType)))
        {
            var probability = NeuralNetworkHelper<double>.GetDefaultLossFunction(task);
            var logits = NeuralNetworkHelper<double>.GetDefaultLogitsLossFunction(task);
            Type expected = probability switch
            {
                CategoricalCrossEntropyLoss<double> => typeof(CrossEntropyWithLogitsLoss<double>),
                BinaryCrossEntropyLoss<double> => typeof(BinaryCrossEntropyWithLogitsLoss<double>),
                DiceLoss<double> => typeof(CompositeLossWithLogits<double>),
                _ => probability.GetType(),
            };
            Assert.True(expected == logits.GetType(), $"{task}: expected {expected.Name}, got {logits.GetType().Name}");
        }
    }

    [Fact]
    public void DenseNetSuppliedSoftmaxHeadUsesProbabilityLossAndHonorsExplicitLoss()
    {
        var architecture = new NeuralNetworkArchitecture<double>(InputType.ThreeDimensional,
            NeuralNetworkTaskType.ImageClassification, inputHeight: 32, inputWidth: 32, inputDepth: 3,
            outputSize: 2, layers: new List<ILayer<double>>
            {
                new DenseLayer<double>(outputSize: 2, activationFunction: new SoftmaxActivation<double>())
            });
        var configuration = new DenseNetConfiguration(DenseNetVariant.Custom, numClasses: 2,
            inputHeight: 32, inputWidth: 32, inputChannels: 3, customBlockLayers: [1]);
        using var network = new DenseNetNetwork<double>(architecture, configuration);
        var loss = Assert.IsType<CategoricalCrossEntropyLoss<double>>(network.DefaultLossFunction);
        Assert.Equal(-Math.Log(0.8), loss.CalculateLoss(new Vector<double>([0.8, 0.2]),
            new Vector<double>([1.0, 0.0])), 10);

        var explicitLoss = new MeanSquaredErrorLoss<double>();
        using var custom = new DenseNetNetwork<double>(architecture.CloneForModelConstruction(), configuration,
            lossFunction: explicitLoss);
        Assert.Same(explicitLoss, custom.DefaultLossFunction);
    }

    [Fact]
    public void SegmentationLogitsUseSigmoidDiceWithCorrectGradient()
    {
        var loss = Assert.IsType<CompositeLossWithLogits<double>>(
            NeuralNetworkHelper<double>.GetDefaultLogitsLossFunction(NeuralNetworkTaskType.ImageSegmentation));
        Assert.Equal(1, loss.TermCount);
        var logits = new Vector<double>([-2.0, 0.25, 1.5]);
        var target = new Vector<double>([0.0, 1.0, 1.0]);
        var probabilities = new Vector<double>([1.0 / (1.0 + Math.Exp(2.0)),
            1.0 / (1.0 + Math.Exp(-0.25)), 1.0 / (1.0 + Math.Exp(-1.5))]);
        double expected = new DiceLoss<double>().CalculateLoss(probabilities, target);
        Assert.Equal(expected, loss.CalculateLoss(logits, target), 12);
        var input = new Tensor<double>([3], logits);
        var labels = new Tensor<double>([3], target);
        using var tape = new GradientTape<double>();
        var objective = loss.ComputeTapeLoss(input, labels);
        Assert.Equal(expected, objective[0], 12);
        var gradient = tape.ComputeGradients(objective, new[] { input })[input];
        const double epsilon = 1e-6;
        for (int i = 0; i < logits.Length; i++)
        {
            double original = logits[i];
            logits[i] = original + epsilon;
            double plus = loss.CalculateLoss(logits, target);
            logits[i] = original - epsilon;
            double minus = loss.CalculateLoss(logits, target);
            logits[i] = original;
            Assert.Equal((plus - minus) / (2.0 * epsilon), gradient[i], 7);
        }
    }

    [Theory]
    [InlineData(NeuralNetworkTaskType.SequenceClassification)]
    [InlineData(NeuralNetworkTaskType.TokenClassification)]
    [InlineData(NeuralNetworkTaskType.NaturalLanguageProcessing)]
    [InlineData(NeuralNetworkTaskType.TextGeneration)]
    [InlineData(NeuralNetworkTaskType.SpeechRecognition)]
    [InlineData(NeuralNetworkTaskType.Translation)]
    [InlineData(NeuralNetworkTaskType.MultiClassClassification)]
    [InlineData(NeuralNetworkTaskType.ImageClassification)]
    public void CategoricalTasksTakeSoftmaxCrossEntropyOnLogits(NeuralNetworkTaskType task)
    {
        Assert.IsType<CrossEntropyWithLogitsLoss<double>>(NeuralNetworkHelper<double>.GetDefaultLogitsLossFunction(task));
    }

    [Fact]
    public void AGeneratedHeadTakesTheLogitsLossAndASuppliedHeadKeepsTheProbabilityLoss()
    {
        var generated = new NeuralNetworkArchitecture<double>(InputType.ThreeDimensional,
            NeuralNetworkTaskType.MultiClassClassification, inputHeight: 32, inputWidth: 32, inputDepth: 3, outputSize: 10);
        Assert.IsType<CrossEntropyWithLogitsLoss<double>>(
            NeuralNetworkHelper<double>.GetDefaultLossForGeneratedLogitsHead(generated));

        var supplied = new NeuralNetworkArchitecture<double>(InputType.ThreeDimensional,
            NeuralNetworkTaskType.MultiClassClassification, inputHeight: 32, inputWidth: 32, inputDepth: 3, outputSize: 10,
            layers: new List<ILayer<double>> { new DenseLayer<double>(outputSize: 10) });
        Assert.IsType<CategoricalCrossEntropyLoss<double>>(
            NeuralNetworkHelper<double>.GetDefaultLossForGeneratedLogitsHead(supplied));
    }
}
