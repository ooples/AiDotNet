using System;
using System.Collections.Generic;
using AiDotNet.Autodiff;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Unit tests for <see cref="DiffusionResBlock{T}.ExportComputationGraph"/>.
/// Covers the full half-block chain emitted for the JIT compiler per
/// github.com/ooples/AiDotNet#1015's "DiffusionResBlock graph export"
/// checklist item.
/// </summary>
public class DiffusionResBlockGraphExportTests
{
    /// <summary>
    /// Basic smoke test: export returns a non-null graph node whose
    /// output value has the expected shape. Catches the most likely
    /// regression (wrong shape / null output).
    /// </summary>
    [Fact]
    public void ExportComputationGraph_ReturnsGraph_WithCorrectOutputShape()
    {
        var block = new DiffusionResBlock<float>(
            inChannels: 8,
            outChannels: 8,
            spatialSize: 8,
            timeEmbedDim: 0);

        var x = MakeInput(new[] { 1, 8, 8, 8 });
        var xNode = TensorOperations<float>.Constant(x, name: "x");

        // Force lazy-init of layer weights by running one eager forward.
        _ = block.Forward(x);

        var graph = block.ExportComputationGraph(new List<ComputationNode<float>> { xNode });

        Assert.NotNull(graph);
        Assert.NotNull(graph.Value);
        Assert.Equal(4, graph.Value.Shape.Length);
        Assert.Equal(1, graph.Value.Shape[0]);
        Assert.Equal(8, graph.Value.Shape[1]); // outChannels
        Assert.Equal(8, graph.Value.Shape[2]);
        Assert.Equal(8, graph.Value.Shape[3]);
    }

    /// <summary>
    /// With a skip-conv (channels differ), the exported graph still
    /// produces the correct output shape — the skip path resizes
    /// input channels to match.
    /// </summary>
    [Fact]
    public void ExportComputationGraph_ChannelChange_UsesSkipConv()
    {
        var block = new DiffusionResBlock<float>(
            inChannels: 4,
            outChannels: 8,  // different → skip_conv is 1x1 Conv
            spatialSize: 4,
            timeEmbedDim: 0);

        var x = MakeInput(new[] { 1, 4, 4, 4 });
        _ = block.Forward(x); // warmup

        var xNode = TensorOperations<float>.Constant(x);
        var graph = block.ExportComputationGraph(new List<ComputationNode<float>> { xNode });

        Assert.NotNull(graph);
        Assert.Equal(8, graph.Value.Shape[1]); // outChannels
    }

    /// <summary>
    /// With a time-embedding node supplied and <c>timeEmbedDim</c> &gt; 0,
    /// the exported graph includes the time-conditioning branch (project,
    /// SiLU, reshape, add). Output shape stays consistent with the
    /// non-time-conditioned path.
    /// </summary>
    [Fact]
    public void ExportComputationGraph_WithTimeEmbed_IncludesTimeConditioning()
    {
        var block = new DiffusionResBlock<float>(
            inChannels: 8,
            outChannels: 8,
            spatialSize: 4,
            timeEmbedDim: 16);

        var x = MakeInput(new[] { 1, 8, 4, 4 });
        var timeEmbed = MakeInput(new[] { 1, 16 });
        _ = block.Forward(x, timeEmbed); // warmup both code paths

        var xNode = TensorOperations<float>.Constant(x);
        var timeNode = TensorOperations<float>.Constant(timeEmbed);
        var graph = block.ExportComputationGraph(
            new List<ComputationNode<float>> { xNode, timeNode });

        Assert.NotNull(graph);
        Assert.Equal(8, graph.Value.Shape[1]);
    }

    /// <summary>
    /// Throws ArgumentException on empty inputNodes — fail-fast rather
    /// than IndexOutOfRange later.
    /// </summary>
    [Fact]
    public void ExportComputationGraph_ThrowsOnEmptyInputs()
    {
        var block = new DiffusionResBlock<float>(
            inChannels: 4, outChannels: 4, spatialSize: 4);
        Assert.Throws<ArgumentException>(() =>
            block.ExportComputationGraph(new List<ComputationNode<float>>()));
    }

    /// <summary>
    /// SupportsJitCompilation must be true now that the export is wired.
    /// </summary>
    [Fact]
    public void SupportsJitCompilation_IsTrue()
    {
        var block = new DiffusionResBlock<float>(
            inChannels: 4, outChannels: 4, spatialSize: 4);
        Assert.True(block.SupportsJitCompilation);
    }

    private static Tensor<float> MakeInput(int[] shape)
    {
        int length = 1;
        foreach (var d in shape) length *= d;
        var data = new float[length];
        for (int i = 0; i < data.Length; i++) data[i] = (i + 1) * 0.01f;
        return new Tensor<float>(data, shape);
    }
}
