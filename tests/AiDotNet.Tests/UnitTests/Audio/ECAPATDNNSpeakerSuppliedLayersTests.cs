using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Audio.Speaker;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Audio;

/// <summary>
/// A stack built by <c>LayerHelper.CreateDefaultECAPATDNNSpeakerLayers</c> is the ECAPA-TDNN graph, not a
/// sequential chain, so a speaker given it must run the encoder's topology.
/// </summary>
public class ECAPATDNNSpeakerSuppliedLayersTests
{
    private const int NumMels = 8;
    private const int EmbeddingDim = 12;

    private static ECAPATDNNSpeakerOptions SmallOptions() => new()
    {
        NumMels = NumMels,
        Channels = [16, 16, 16, 48],
        KernelSizes = [5, 3, 3, 1],
        Dilations = [1, 2, 3, 1],
        Res2NetScale = 4,
        SEBottleneckDim = 8,
        AttentionChannels = 8,
        EmbeddingDim = EmbeddingDim
    };

    private static NeuralNetworkArchitecture<double> Architecture(List<ILayer<double>>? layers = null) => new(
        InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: NumMels, outputSize: EmbeddingDim,
        layers: layers);

    private static Tensor<double> Features()
    {
        var features = new Tensor<double>(new[] { 20, NumMels });
        for (int i = 0; i < features.Length; i++) features[i] = Math.Sin(0.31 * i) + 0.1 * Math.Cos(1.7 * i);
        return features;
    }

    [Fact]
    public void FactoryStack_RunsTheEncoderGraph_ToOneEmbedding()
    {
        var layers = LayerHelper<double>.CreateDefaultECAPATDNNSpeakerLayers(
            numMels: NumMels, channels: 16, embeddingDim: EmbeddingDim, numBlocks: 2, poolingDim: 48,
            seBottleneckDim: 8, res2NetScale: 4, attentionChannels: 8).ToList();
        var model = new ECAPATDNNSpeaker<double>(Architecture(layers), SmallOptions());
        model.SetTrainingMode(false);

        var embedding = model.Predict(Features());

        Assert.Equal(new[] { EmbeddingDim }, embedding.Shape.ToArray());
        Assert.All(embedding.ToArray(), value => Assert.False(double.IsNaN(value) || double.IsInfinity(value)));
    }

    [Fact]
    public void FactoryStack_ComputesWhatTheDefaultModelComputes_WithTheSameWeights()
    {
        var layers = LayerHelper<double>.CreateDefaultECAPATDNNSpeakerLayers(
            numMels: NumMels, channels: 16, embeddingDim: EmbeddingDim, numBlocks: 2, poolingDim: 48,
            seBottleneckDim: 8, res2NetScale: 4, attentionChannels: 8).ToList();
        var supplied = new ECAPATDNNSpeaker<double>(Architecture(layers), SmallOptions());
        supplied.SetTrainingMode(false);
        var actual = supplied.Predict(Features()).ToArray();

        var reference = new ECAPATDNNSpeaker<double>(Architecture(), SmallOptions());
        reference.SetTrainingMode(false);
        _ = reference.Predict(Features());
        var weights = supplied.GetParameters();
        Assert.Equal(weights.Length, reference.GetParameters().Length);
        reference.SetParameters(weights);

        Assert.Equal(reference.Predict(Features()).ToArray(), actual);
    }
}
