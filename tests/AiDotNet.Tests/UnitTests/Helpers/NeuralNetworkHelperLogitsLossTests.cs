using System;
using System.Collections.Generic;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Helpers;

/// <summary>
/// The default loss for a raw-logits head is the with-logits form of the probability default, for every task type, and a
/// network keeps the probability default for a head the caller supplied.
/// </summary>
public class NeuralNetworkHelperLogitsLossTests
{
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
                _ => probability.GetType(),
            };
            Assert.True(expected == logits.GetType(), $"{task}: expected {expected.Name}, got {logits.GetType().Name}");
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
