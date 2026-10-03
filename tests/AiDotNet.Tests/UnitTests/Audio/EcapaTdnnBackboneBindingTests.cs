using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Audio;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Audio;

/// <summary>
/// ECAPA-TDNN binds a model's layer list to its roles by position after a deserialize or clone. A list
/// with the right layer types but a stale convolution width must be refused, and the refusal must leave
/// the encoder on its current graph rather than half-bound.
/// </summary>
public class EcapaTdnnBackboneBindingTests
{
    private static EcapaTdnnBackbone<double> CreateBackbone() => new(
        channels: new[] { 8, 8, 8 },
        kernelSizes: new[] { 5, 3, 1 },
        dilations: new[] { 1, 2, 1 },
        res2NetScale: 2,
        seChannels: 4,
        attentionChannels: 4,
        embeddingDimension: 4);

    [Fact]
    public void BindTo_AcceptsAFreshCopyOfTheLayout()
    {
        var backbone = CreateBackbone();
        var replacement = CreateBackbone().Layers.ToList();

        backbone.BindTo(replacement);

        Assert.True(replacement.SequenceEqual(backbone.Layers));
    }

    [Fact]
    public void BindTo_RefusesAConvolutionWithTheWrongWidth_AndKeepsTheCurrentGraph()
    {
        var backbone = CreateBackbone();
        var original = backbone.Layers.ToList();
        var replacement = CreateBackbone().Layers.ToList();
        int convolution = replacement.FindIndex(layer => layer is Conv1DLayer<double>);
        replacement[convolution] = new Conv1DLayer<double>(16, 5);

        var error = Assert.Throws<InvalidOperationException>(() => backbone.BindTo(replacement));

        Assert.Contains("OutputChannels", error.Message);
        Assert.True(original.SequenceEqual(backbone.Layers));
    }
}