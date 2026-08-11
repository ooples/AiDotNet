using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Finance.NLP;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

public class BertEmbeddingLayerTests
{
    [Fact]
    public async Task Domain_IsWordVocabulary_AndOutputShapeAppendsHiddenSize()
    {
        await Task.Yield();

        using var layer = NewLayer();
        layer.SetTrainingMode(false);
        var input = TokenIds([1, 4], 3, 7, 11, 13);

        var output = layer.Forward(input);

        var domain = layer.GetInputDomain([1, 4]);
        Assert.True(domain.IsIndices);
        Assert.Equal(32, domain.MaxExclusive);
        Assert.Equal(new[] { 1, 4, 8 }, output.Shape);
    }

    [Fact]
    public async Task OptionalTokenTypeIds_UseParallelLookupWithoutChangingShape()
    {
        await Task.Yield();

        using var layer = NewLayer();
        layer.SetTrainingMode(false);
        var input = TokenIds([1, 4], 3, 7, 11, 13);
        var segmentZero = TokenIds([1, 4], 0, 0, 0, 0);
        var segmentOne = TokenIds([1, 4], 1, 1, 1, 1);

        var zeroOutput = layer.Forward(input, segmentZero);
        var oneOutput = layer.Forward(input, segmentOne);

        Assert.Equal(zeroOutput.Shape, oneOutput.Shape);
        Assert.True(AnyDifferent(zeroOutput, oneOutput),
            "Changing token-type IDs should change the parallel token-type embedding contribution.");
    }

    [Fact]
    public async Task Parameters_AreDiscoveredRecursivelyByGeneratedSubLayerRegistration()
    {
        await Task.Yield();

        using var layer = NewLayer();
        layer.SetTrainingMode(false);
        _ = layer.Forward(TokenIds([1, 2], 1, 2));

        // word + learned position + token type + LayerNorm gamma/beta
        int expected = (32 * 8) + (16 * 8) + (2 * 8) + (2 * 8);
        Assert.Equal(expected, layer.ParameterCount);
        Assert.Equal(expected, layer.GetParameters().Length);
        Assert.Equal(5, layer.GetSubLayers().Count);
    }

    [Fact]
    public async Task InvalidTokenTypeId_IsRejectedByItsOwnLookupDomain()
    {
        await Task.Yield();

        using var layer = NewLayer();
        var input = TokenIds([1, 2], 1, 2);
        var invalidTypes = TokenIds([1, 2], 0, 2);

        var ex = Assert.Throws<ArgumentException>(() => layer.Forward(input, invalidTypes));

        Assert.Contains("[0, 2)", ex.Message);
    }

    [Fact]
    public async Task RankThreeTokenIds_AreRejectedByTheDeclaredShapeContract()
    {
        await Task.Yield();

        using var layer = NewLayer();
        var legacySingletonFeatureAxis = TokenIds([1, 2, 1], 1, 2);

        var ex = Assert.Throws<ArgumentException>(() => layer.Forward(legacySingletonFeatureAxis));

        Assert.Contains("expects [sequence] or [batch, sequence]", ex.Message);
        Assert.Contains("got [1, 2, 1]", ex.Message);
    }

    [Fact]
    public async Task SecBertBuilder_UsesOneCompositeEmbeddingBlock()
    {
        await Task.Yield();

        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 4,
            outputSize: 1);
        var layers = LayerHelper<double>.CreateDefaultSECBERTLayers(
            architecture,
            maxSequenceLength: 16,
            vocabularySize: 32,
            hiddenSize: 8,
            numAttentionHeads: 2,
            numHiddenLayers: 1,
            dropoutProbability: 0.0).ToList();

        Assert.IsType<BertEmbeddingLayer<double>>(layers[0]);
        Assert.Single(layers.OfType<BertEmbeddingLayer<double>>());
        Assert.Empty(layers.OfType<PositionalEncodingLayer<double>>());
        Assert.Empty(layers.OfType<EmbeddingLayer<double>>());
    }

    [Fact]
    public async Task SecBertOptionsBuilder_UsesTheSameCanonicalTopology()
    {
        await Task.Yield();

        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: 4,
            outputSize: 3);
        var layers = LayerHelper<double>.CreateDefaultSECBERTLayers(
            architecture,
            vocabularySize: 32,
            maxSequenceLength: 16,
            hiddenDimension: 8,
            numAttentionHeads: 2,
            intermediateDimension: 24,
            numLayers: 1,
            numClasses: 3,
            dropoutRate: 0.0,
            taskType: "classification").ToList();

        Assert.IsType<BertEmbeddingLayer<double>>(layers[0]);
        Assert.Single(layers.OfType<BertEmbeddingLayer<double>>());
        Assert.Single(layers.OfType<TransformerEncoderBlock<double>>());
        Assert.Empty(layers.OfType<EmbeddingLayer<double>>());
        Assert.Empty(layers.OfType<MultiHeadAttentionLayer<double>>());
    }

    [Fact]
    public async Task FinancialBertOptionsBuilder_UsesTheSameCanonicalTopology()
    {
        await Task.Yield();

        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: 4,
            outputSize: 3);
        var layers = LayerHelper<double>.CreateDefaultFinancialBERTLayers(
            architecture,
            vocabularySize: 32,
            maxSequenceLength: 16,
            hiddenDimension: 8,
            numAttentionHeads: 2,
            intermediateDimension: 24,
            numLayers: 1,
            numClasses: 3,
            dropoutRate: 0.0,
            taskType: "sentiment").ToList();

        Assert.IsType<BertEmbeddingLayer<double>>(layers[0]);
        Assert.Single(layers.OfType<BertEmbeddingLayer<double>>());
        Assert.Single(layers.OfType<TransformerEncoderBlock<double>>());
        Assert.Empty(layers.OfType<EmbeddingLayer<double>>());
        Assert.Empty(layers.OfType<MultiHeadAttentionLayer<double>>());
    }

    [Fact]
    public async Task FinBertToneOptionsBuilder_UsesCanonicalCompositeAndResidualBlock()
    {
        await Task.Yield();

        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: 4,
            outputSize: 3);
        var layers = LayerHelper<double>.CreateDefaultFinBERTToneLayers(
            architecture,
            vocabularySize: 32,
            maxSequenceLength: 16,
            hiddenDimension: 8,
            numAttentionHeads: 2,
            intermediateDimension: 24,
            numLayers: 1,
            numToneClasses: 3,
            dropoutRate: 0.0).ToList();

        Assert.IsType<BertEmbeddingLayer<double>>(layers[0]);
        Assert.Single(layers.OfType<BertEmbeddingLayer<double>>());
        Assert.Single(layers.OfType<TransformerEncoderBlock<double>>());
        Assert.Empty(layers.OfType<EmbeddingLayer<double>>());
        Assert.Empty(layers.OfType<MultiHeadAttentionLayer<double>>());
    }

    [Fact]
    public async Task FinBertTone_CloneAfterTraining_PreservesParametersAndOutput()
    {
        await Task.Yield();

        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: 8,
            outputSize: 3)
        {
            RandomSeed = 1337
        };
        var options = new FinBERTToneOptions<double>
        {
            MaxSequenceLength = 8,
            VocabularySize = 32,
            HiddenDimension = 8,
            NumAttentionHeads = 2,
            IntermediateDimension = 16,
            NumLayers = 1,
            NumToneClasses = 3,
            DropoutRate = 0.0
        };
        using var network = new FinBERTTone<double>(architecture, options);
        var input = TokenIds([1, 8], 1, 2, 3, 4, 5, 6, 7, 8);
        var initialOutput = network.Predict(input);
        var target = new Tensor<double>(initialOutput.Shape.ToArray());
        target[0] = 1.0;

        network.Train(input, target);
        foreach (var layer in network.Layers) layer.ResetState();
        var trainedOutput = network.Predict(input);
        var trainedParameters = network.GetParameters();

        using var clone = Assert.IsType<FinBERTTone<double>>(network.Clone());
        foreach (var layer in clone.Layers) layer.ResetState();
        var clonedParameters = clone.GetParameters();
        var clonedOutput = clone.Predict(input);

        Assert.Equal(trainedParameters.Length, clonedParameters.Length);
        for (int i = 0; i < trainedParameters.Length; i++)
        {
            Assert.True(trainedParameters[i] == clonedParameters[i],
                $"Parameter {i} differs: trained={trainedParameters[i]:G17}, clone={clonedParameters[i]:G17}.");
        }

        var trainedEmbedded = network.Layers[0].Forward(input);
        var clonedEmbedded = clone.Layers[0].Forward(input);
        AssertTensorsEqual("embedding", trainedEmbedded, clonedEmbedded);
        var trainedBlock = Assert.IsType<TransformerEncoderBlock<double>>(network.Layers[1]);
        var clonedBlock = Assert.IsType<TransformerEncoderBlock<double>>(clone.Layers[1]);
        var trainedChildren = trainedBlock.GetSubLayers();
        var clonedChildren = clonedBlock.GetSubLayers();
        Assert.Equal(trainedChildren.Count, clonedChildren.Count);
        for (int childIndex = 0; childIndex < trainedChildren.Count; childIndex++)
        {
            Assert.Equal(trainedChildren[childIndex].GetType(), clonedChildren[childIndex].GetType());
            var trainedChildParameters = trainedChildren[childIndex].GetParameters();
            var clonedChildParameters = clonedChildren[childIndex].GetParameters();
            Assert.Equal(trainedChildParameters.Length, clonedChildParameters.Length);
            for (int i = 0; i < trainedChildParameters.Length; i++)
            {
                Assert.True(trainedChildParameters[i] == clonedChildParameters[i],
                    $"Child {childIndex} parameter {i} differs.");
            }
        }
        var trainedNorm1 = trainedChildren[1].Forward(trainedEmbedded);
        var clonedNorm1 = clonedChildren[1].Forward(clonedEmbedded);
        AssertTensorsEqual("norm1", trainedNorm1, clonedNorm1);
        var trainedAttention = trainedChildren[0].Forward(trainedNorm1);
        var clonedAttention = clonedChildren[0].Forward(clonedNorm1);
        AssertTensorsEqual("attention", trainedAttention, clonedAttention);

        var trainedActivations = network.GetNamedLayerActivations(input);
        var clonedActivations = clone.GetNamedLayerActivations(input);
        foreach (var (name, trainedActivation) in trainedActivations)
        {
            Assert.True(clonedActivations.TryGetValue(name, out var clonedActivation),
                $"Clone is missing activation {name}.");
            Assert.Equal(trainedActivation.Shape, clonedActivation.Shape);
            for (int i = 0; i < trainedActivation.Length; i++)
            {
                Assert.True(Math.Abs(trainedActivation[i] - clonedActivation[i]) < 1e-12,
                    $"Activation {name}[{i}] differs: trained={trainedActivation[i]:G17}, clone={clonedActivation[i]:G17}.");
            }
        }

        Assert.Equal(trainedOutput.Shape, clonedOutput.Shape);
        for (int i = 0; i < trainedOutput.Length; i++)
        {
            Assert.True(Math.Abs(trainedOutput[i] - clonedOutput[i]) < 1e-12,
                $"Output {i} differs: trained={trainedOutput[i]:G17}, clone={clonedOutput[i]:G17}.");
        }
    }

    private static BertEmbeddingLayer<double> NewLayer() =>
        new(vocabularySize: 32, hiddenSize: 8, maxSequenceLength: 16,
            tokenTypeVocabularySize: 2, dropoutProbability: 0.0);

    private static Tensor<double> TokenIds(int[] shape, params double[] values) =>
        new(new Vector<double>(values), shape);

    private static bool AnyDifferent(Tensor<double> left, Tensor<double> right)
    {
        for (int i = 0; i < left.Length; i++)
        {
            if (Math.Abs(left[i] - right[i]) > 1e-12)
                return true;
        }

        return false;
    }

    private static void AssertTensorsEqual(string name, Tensor<double> expected, Tensor<double> actual)
    {
        Assert.Equal(expected.Shape, actual.Shape);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(Math.Abs(expected[i] - actual[i]) < 1e-12,
                $"{name}[{i}] differs: expected={expected[i]:G17}, actual={actual[i]:G17}.");
        }
    }
}
