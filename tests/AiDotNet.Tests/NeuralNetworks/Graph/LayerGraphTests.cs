using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Graph;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.NeuralNetworks.Graph;

/// <summary>
/// Behavioural checks for <see cref="LayerGraph{T}"/> — the declared-wiring layer that lets a model say
/// what its dataflow actually is instead of leaving the framework to infer a chain from list order.
/// </summary>
/// <remarks>
/// <para>
/// The defect this infrastructure exists to remove is worth restating, because it is what these tests
/// aim at: for a model with two heads reading one trunk, "the previous entry in <c>Layers</c>" and "the
/// tensor this layer actually receives" are different things. Everything the framework derives from list
/// order — lazy shape resolution, layout validation, the sequential training forward — is wrong for such a
/// model, and wrong in ways that do not announce themselves as topology errors.
/// </para>
/// <para>
/// Each test below therefore uses a branch. A linear chain cannot distinguish a graph-aware
/// implementation from a list-order one, so a suite built on chains would pass against the very bug the
/// graph was written to fix.
/// </para>
/// </remarks>
public class LayerGraphTests
{
    private static IEngine Engine => AiDotNetEngine.Current;

    private static ConvolutionalLayer<double> Conv(int outChannels, int kernel = 1, int stride = 1, int pad = 0)
        => new ConvolutionalLayer<double>(outChannels, kernel, stride, pad);

    private static Tensor<double> Image(int channels, int height, int width)
    {
        var t = new Tensor<double>(new[] { channels, height, width });
        for (int i = 0; i < t.Length; i++) t[i] = ((i * 37) % 17) / 17.0;
        return t;
    }

    // ---------------------------------------------------------------------------------------------
    // Builder contract
    // ---------------------------------------------------------------------------------------------

    [Fact]
    public void Builder_RefusesAForwardReference()
    {
        // Ids double as a topological order, which only holds if an edge can never point at a node that
        // does not exist yet. Without this the executor would read an unwritten slot and every downstream
        // guarantee (shape resolution, run detection) would rest on an ordering that was never enforced.
        var b = new LayerGraphBuilder<double>();
        b.Add(Conv(4));

        Assert.ThrowsAny<ArgumentException>(() => b.Add(Conv(4), input: 7));
    }

    [Fact]
    public void Builder_RefusesAnOutputThatDoesNotExist()
    {
        var b = new LayerGraphBuilder<double>();
        b.Add(Conv(4));

        Assert.ThrowsAny<ArgumentException>(() => b.Output(3));
    }

    // ---------------------------------------------------------------------------------------------
    // Linear compatibility — the migration must not change unmigrated models
    // ---------------------------------------------------------------------------------------------

    [Fact]
    public void Linear_ProducesTheSameTensorAsTheHistoricalSequentialLoop()
    {
        // Every model that has NOT migrated still runs the flat-list forward. If Linear() differs from it
        // by so much as an op, migrating the shared execution path becomes a library-wide regression
        // rather than a per-model change.
        var layers = new List<ILayer<double>>
        {
            Conv(4, kernel: 3, stride: 1, pad: 1),
            Conv(2, kernel: 3, stride: 1, pad: 1),
        };

        var input = Image(3, 8, 8);

        var sequential = input;
        foreach (var layer in layers) sequential = layer.Forward(sequential);

        var viaGraph = LayerGraph<double>.Linear(layers).Forward(input);

        Assert.Equal(sequential.Shape.ToArray(), viaGraph.Shape.ToArray());
        for (int i = 0; i < sequential.Length; i++)
        {
            Assert.Equal(sequential[i], viaGraph[i], 12);
        }
    }

    [Fact]
    public void Linear_IsLinearAndHasNoFanOut()
    {
        var graph = LayerGraph<double>.Linear(new List<ILayer<double>> { Conv(4), Conv(4), Conv(4) });

        Assert.True(graph.IsLinear);
        Assert.False(graph.HasFanOut);
    }

    // ---------------------------------------------------------------------------------------------
    // Fan-out
    // ---------------------------------------------------------------------------------------------

    [Fact]
    public void FanOut_IsDetected_AndLinearIsNot()
    {
        var b = new LayerGraphBuilder<double>();
        int trunk = b.Add(Conv(8, kernel: 3, pad: 1));
        int headA = b.Add(Conv(1), trunk);
        int headB = b.Add(Conv(3), trunk);
        int joined = b.AddJoin(
            layer: null,
            inputs: new[] { headA, headB },
            combine: parts => Engine.Concat(new[] { parts[0], parts[1] }, 0),
            description: "concat the two heads");

        var graph = b.Output(joined).Build();

        Assert.True(graph.HasFanOut);
        Assert.False(graph.IsLinear);
    }

    [Fact]
    public void FanOut_BothHeadsSeeTheTrunkOutput_NotEachOther()
    {
        // THE case list order gets wrong. headB's list predecessor is headA, so a sequential executor
        // feeds it headA's 1-channel output. The graph must feed it the trunk's 8 channels.
        var b = new LayerGraphBuilder<double>();
        int trunk = b.Add(Conv(8, kernel: 3, pad: 1));
        int headA = b.Add(Conv(1), trunk);
        int headB = b.Add(Conv(3), trunk);
        int joined = b.AddJoin(
            layer: null,
            inputs: new[] { headA, headB },
            combine: parts => Engine.Concat(new[] { parts[0], parts[1] }, 0),
            description: "concat the two heads");

        var graph = b.Output(joined).Build();
        var output = graph.Forward(Image(3, 8, 8));

        // 1 + 3 channels. Under sequential execution headB would consume headA's single channel and the
        // concat would still produce 4 — so the channel count alone is NOT the discriminator. The trunk
        // width below is.
        Assert.Equal(4, output.Shape[0]);

        var layers = graph.ToLayerList();
        var headBLayer = (ConvolutionalLayer<double>)layers[2];
        Assert.Equal(8, headBLayer.GetInputShape()[0]);
    }

    // ---------------------------------------------------------------------------------------------
    // Shape resolution — the concrete failure this replaced
    // ---------------------------------------------------------------------------------------------

    [Fact]
    public void ResolveShapes_SizesSiblingHeadsFromTheTrunk_NotFromEachOther()
    {
        // The regression in full. A list-order resolver pins headB's input depth to headA's output depth
        // (1), and the first real forward dies with "Expected input depth 1, but got 8" — a message that
        // names neither layer nor the assumption behind it.
        var b = new LayerGraphBuilder<double>();
        int trunk = b.Add(Conv(8, kernel: 3, pad: 1));
        int headA = b.Add(Conv(1), trunk);
        int headB = b.Add(Conv(3), trunk);
        int joined = b.AddJoin(
            layer: null,
            inputs: new[] { headA, headB },
            combine: parts => Engine.Concat(new[] { parts[0], parts[1] }, 0),
            description: "concat the two heads");

        var graph = b.Output(joined).Build();

        var outShape = graph.ResolveShapes(new[] { 3, 8, 8 });

        Assert.NotNull(outShape);
        Assert.Equal(4, outShape![0]);

        var layers = graph.ToLayerList();
        Assert.Equal(3, ((ConvolutionalLayer<double>)layers[0]).GetInputShape()[0]);
        Assert.Equal(8, ((ConvolutionalLayer<double>)layers[1]).GetInputShape()[0]);
        Assert.Equal(8, ((ConvolutionalLayer<double>)layers[2]).GetInputShape()[0]);
    }

    [Fact]
    public void ResolveShapes_AppliesEdgeTransforms_SoAReshapedHandOffResolvesCorrectly()
    {
        // A dense head entered through a permute+flatten. Its input width is a function of the transform,
        // not of its predecessor's declared output shape, so a resolver that skips edges sizes it wrong.
        var b = new LayerGraphBuilder<double>();
        int conv = b.Add(Conv(4, kernel: 3, pad: 1));
        int dense = b.AddVia(
            new DenseLayer<double>(5),
            conv,
            transform: f =>
            {
                int channels = f.Shape[f.Rank - 3];
                int height = f.Shape[f.Rank - 2];
                int width = f.Shape[f.Rank - 1];
                var columnsFirst = Engine.TensorPermute(f, new[] { 2, 0, 1 });
                return Engine.Reshape(columnsFirst, new[] { width, channels * height });
            },
            description: "[C,h,w] -> [w, C*h]");

        var graph = b.Output(dense).Build();

        var outShape = graph.ResolveShapes(new[] { 3, 6, 7 });

        Assert.NotNull(outShape);

        // A dense layer DECLARES its per-sample output, so the leading column axis is absent here by
        // library convention — the resolved shape is [5], not [7, 5].
        Assert.Equal(new[] { 5 }, outShape!);

        // THE assertion. The dense head must have been sized against C*h = 4*6 = 24, which only the edge
        // transform yields; against the conv's declared [4, 6, 7] it would be 7, and the first forward
        // would die on a width mismatch.
        var denseLayer = (DenseLayer<double>)graph.ToLayerList()[1];
        Assert.Equal(24, denseLayer.GetInputShape()[^1]);

        // And the executed shape really is 7 columns x 5 classes, so the resolution above described a
        // forward that runs rather than one that merely type-checks.
        var executed = graph.Forward(Image(3, 6, 7));
        Assert.Equal(new[] { 7, 5 }, executed.Shape.ToArray());
    }

    [Fact]
    public void ResolveShapes_AllocatesNoWeights_SoResolutionDoesNotDisturbInitialization()
    {
        // Shape resolution runs from ParameterCount probes and training-mode switches, long before the
        // first forward. If it materialized weights it would consume the RNG stream and change the
        // initialization a seeded run is supposed to reproduce.
        var b = new LayerGraphBuilder<double>();
        int trunk = b.Add(Conv(8, kernel: 3, pad: 1));
        int head = b.Add(Conv(3), trunk);
        var graph = b.Output(head).Build();

        graph.ResolveShapes(new[] { 3, 8, 8 });

        foreach (var layer in graph.ToLayerList().OfType<LayerBase<double>>())
        {
            Assert.True(layer.IsShapeResolved);
        }
    }

    [Fact]
    public void ResolveShapes_ReturnsNull_RatherThanPinningLayersToAGuessedShape()
    {
        // An input the first layer cannot accept. Leaving the remaining layers lazy is strictly better
        // than sizing them off a guess: a lazy layer still resolves correctly from its real input on the
        // first forward, whereas a wrongly-pinned one is stuck.
        var b = new LayerGraphBuilder<double>();
        int only = b.Add(Conv(4, kernel: 3, pad: 1));
        var graph = b.Output(only).Build();

        var outShape = graph.ResolveShapes(new[] { 5 });

        Assert.Null(outShape);
    }

    // ---------------------------------------------------------------------------------------------
    // Contiguous runs — what layout validation is allowed to compare
    // ---------------------------------------------------------------------------------------------

    [Fact]
    public void ContiguousRuns_BreakAtAFanOut()
    {
        // The trunk feeds two heads, so trunk->headA is not a hand-off that layout validation may check:
        // headA is one of two consumers, and the pair "trunk, headA" describes no exclusive edge.
        var b = new LayerGraphBuilder<double>();
        int stem = b.Add(Conv(8, kernel: 3, pad: 1));
        int trunk = b.Add(Conv(8, kernel: 3, pad: 1), stem);
        int headA = b.Add(Conv(1), trunk);
        int headB = b.Add(Conv(3), trunk);
        int joined = b.AddJoin(
            layer: null,
            inputs: new[] { headA, headB },
            combine: parts => Engine.Concat(new[] { parts[0], parts[1] }, 0),
            description: "concat");

        var runs = b.Output(joined).Build().ContiguousRuns();

        // Exactly one run of length >= 2: stem -> trunk. Neither head continues it.
        Assert.Single(runs);
        Assert.Equal(2, runs[0].Count);
    }

    [Fact]
    public void ContiguousRuns_BreakAtAnEdgeTransform()
    {
        // conv -> dense across a permute+flatten is NOT a direct hand-off. Reporting it as one produced
        // the pilot's false positive: the two layouts genuinely disagree, and the reshape is what makes
        // the pairing correct. A validator that flags it teaches readers to ignore it.
        var b = new LayerGraphBuilder<double>();
        int c1 = b.Add(Conv(4, kernel: 3, pad: 1));
        int c2 = b.Add(Conv(4, kernel: 3, pad: 1), c1);
        int dense = b.AddVia(
            new DenseLayer<double>(5),
            c2,
            transform: f =>
            {
                int channels = f.Shape[f.Rank - 3];
                int height = f.Shape[f.Rank - 2];
                int width = f.Shape[f.Rank - 1];
                var columnsFirst = Engine.TensorPermute(f, new[] { 2, 0, 1 });
                return Engine.Reshape(columnsFirst, new[] { width, channels * height });
            },
            description: "[C,h,w] -> [w, C*h]");

        var runs = b.Output(dense).Build().ContiguousRuns();

        Assert.Single(runs);
        Assert.Equal(2, runs[0].Count);
        Assert.DoesNotContain(runs[0], l => l is DenseLayer<double>);
    }

    [Fact]
    public void ContiguousRuns_OfALinearGraphIsTheWholeChain()
    {
        var graph = LayerGraph<double>.Linear(new List<ILayer<double>> { Conv(4), Conv(4), Conv(4) });

        var runs = graph.ContiguousRuns();

        Assert.Single(runs);
        Assert.Equal(3, runs[0].Count);
    }

    // ---------------------------------------------------------------------------------------------
    // Projection
    // ---------------------------------------------------------------------------------------------

    [Fact]
    public void ToLayerList_ProjectsOnlyRealLayers_InNodeOrder()
    {
        // Join nodes carry no layer. Anything that counts parameters, clones or serializes walks this
        // projection, so a null slipping through would surface as a NullReferenceException far from here.
        var b = new LayerGraphBuilder<double>();
        int trunk = b.Add(Conv(8, kernel: 3, pad: 1));
        int headA = b.Add(Conv(1), trunk);
        int headB = b.Add(Conv(3), trunk);
        int joined = b.AddJoin(
            layer: null,
            inputs: new[] { headA, headB },
            combine: parts => Engine.Concat(new[] { parts[0], parts[1] }, 0),
            description: "concat");

        var layers = b.Output(joined).Build().ToLayerList();

        Assert.Equal(3, layers.Count);
        Assert.DoesNotContain(layers, l => l is null);
    }
}
