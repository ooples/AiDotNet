using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ComputerVision.OCR.EndToEnd;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Graph;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.NeuralNetworks.Graph;

/// <summary>
/// Proves a model's dataflow can be RECOVERED from one forward pass, with no declaration.
/// </summary>
/// <remarks>
/// <para>
/// The point of tracing is to stop asking models to declare their topology. Chain validation reading
/// <c>Layers</c> as a linear list pairs layers that never meet — measured across every constructible
/// model, that produced four reports and not one was a real defect, all four being branched models whose
/// branches had been flattened into a single list.
/// </para>
/// <para>
/// ABCNET IS THE REFERENCE ANSWER, and that is why it is the model tested here. It already declares its
/// wiring, so there is a known-correct graph to check the tracer against rather than a guess. When the
/// traced graph agrees with the declared one, the declaration becomes redundant — which is the whole
/// argument for tracing over annotation.
/// </para>
/// </remarks>
public class LayerForwardTracingTests
{
    private readonly ITestOutputHelper _out;
    public LayerForwardTracingTests(ITestOutputHelper output) => _out = output;

    private static ABCNet<double> Model() =>
        new(new ABCNetOptions<double> { InputHeight = 32, InputWidth = 32, FeatureChannels = 16 });

    private static Tensor<double> Image(int c, int h, int w)
    {
        var t = new Tensor<double>(new[] { c, h, w });
        for (int i = 0; i < t.Length; i++) t[i] = ((i * 23) % 19) / 19.0;
        return t;
    }

    [Fact]
    public void ObserverRecordsNothingWhenNotTracing()
    {
        // The wrapper must be inert by default. A framework that records on every forward in production
        // has bought diagnostics with a permanent tax.
        Assert.Null(LayerForwardObserver<double>.Current);

        var layer = new ConvolutionalLayer<double>(outputDepth: 4, kernelSize: 3, stride: 1, padding: 1);
        layer.Forward(Image(3, 8, 8));

        Assert.Null(LayerForwardObserver<double>.Current);
    }

    [Fact]
    public void TracingRecordsEveryMigratedLayerCall()
    {
        using var trace = new LayerForwardObserver<double>();
        var layer = new ConvolutionalLayer<double>(outputDepth: 4, kernelSize: 3, stride: 1, padding: 1);
        var input = Image(3, 8, 8);

        var output = layer.Forward(input);

        Assert.Single(trace.Calls);
        Assert.Same(layer, trace.Calls[0].Layer);
        Assert.Same(input, trace.Calls[0].Input);
        Assert.Same(output, trace.Calls[0].Output);
    }

    [Fact]
    public void EdgesComeFromTensorIdentity_SoAChainIsRecoveredWithoutBeingDeclared()
    {
        using var trace = new LayerForwardObserver<double>();
        var first = new ConvolutionalLayer<double>(outputDepth: 4, kernelSize: 3, stride: 1, padding: 1);
        var second = new ConvolutionalLayer<double>(outputDepth: 2, kernelSize: 3, stride: 1, padding: 1);

        var mid = first.Forward(Image(3, 8, 8));
        second.Forward(mid);

        var producers = trace.ResolveProducers();

        Assert.Equal(-1, producers[0]);   // came from outside the traced layers
        Assert.Equal(0, producers[1]);    // second consumed first's output
        Assert.False(trace.HasFanOut);
    }

    [Fact]
    public void FanOutIsDetectedWithoutAnyDeclaration()
    {
        // One tensor feeding two layers. A flat list cannot express this; identity makes it obvious.
        using var trace = new LayerForwardObserver<double>();
        var trunk = new ConvolutionalLayer<double>(outputDepth: 8, kernelSize: 3, stride: 1, padding: 1);
        var headA = new ConvolutionalLayer<double>(outputDepth: 1, kernelSize: 1, stride: 1, padding: 0);
        var headB = new ConvolutionalLayer<double>(outputDepth: 3, kernelSize: 1, stride: 1, padding: 0);

        var features = trunk.Forward(Image(3, 8, 8));
        headA.Forward(features);
        headB.Forward(features);

        var producers = trace.ResolveProducers();

        Assert.Equal(0, producers[1]);
        Assert.Equal(0, producers[2]);   // BOTH heads read the trunk, not each other
        Assert.True(trace.HasFanOut);

        // And the runs break at the branch, so no validator pairs headA with headB.
        Assert.DoesNotContain(trace.ContiguousRuns(), run => run.Contains(headA) && run.Contains(headB));
    }

    // ---------------------------------------------------------------------------------------------
    // The real test: does tracing recover what ABCNet declares?
    // ---------------------------------------------------------------------------------------------

    [Fact]
    public void TracingABCNet_RecoversTheFanOutItDeclares()
    {
        var model = Model();

        using var trace = new LayerForwardObserver<double>();
        model.Predict(Image(3, 32, 32));

        _out.WriteLine($"recorded {trace.Calls.Count} layer calls");
        var producers = trace.ResolveProducers();
        for (int i = 0; i < trace.Calls.Count; i++)
        {
            _out.WriteLine($"  [{i}] {trace.Calls[i].Layer.GetType().Name} <- {producers[i]}");
        }

        Assert.True(trace.Calls.Count >= 2, "no migrated layers were recorded at all");

        // ABCNet's detection path fans out: the trunk feeds BOTH the score head and the Bezier head.
        // That is the structural fact its declared graph asserts, recovered here from execution alone.
        Assert.True(
            trace.HasFanOut,
            "tracing ABCNet's detection path did not observe the fan-out its declared graph asserts. "
            + $"Producers: [{string.Join(", ", producers)}]");
    }

    [Fact]
    public void TracedRunsNeverPairTwoSiblingHeads()
    {
        // The agreement that makes the declaration redundant: whatever the tracer groups into a run, it
        // must never place two layers that read the SAME tensor next to each other. That is the exact
        // false pairing the linear reading of Layers produced on all four branched models.
        //
        // Compared by REFERENCE, not by type name. ABCNet's layers are all ConvolutionalLayer, so a
        // name-based comparison cannot tell the stem->mid hand-off from the score/Bezier sibling pair -
        // it reads "Conv->Conv" either way. Identity is the only thing that distinguishes them, which is
        // the same reason the tracer itself resolves edges by tensor identity rather than by ordering.
        var model = Model();

        using var trace = new LayerForwardObserver<double>();
        model.Predict(Image(3, 32, 32));

        var producers = trace.ResolveProducers();
        var runs = trace.ContiguousRuns();

        for (int i = 1; i < trace.Calls.Count; i++)
        {
            if (producers[i] < 0 || producers[i] != producers[i - 1]) continue;

            var siblingA = trace.Calls[i - 1].Layer;
            var siblingB = trace.Calls[i].Layer;
            _out.WriteLine($"siblings at calls {i - 1},{i} both read producer {producers[i]}");

            foreach (var run in runs)
            {
                for (int k = 0; k + 1 < run.Count; k++)
                {
                    bool adjacent =
                        ReferenceEquals(run[k], siblingA) && ReferenceEquals(run[k + 1], siblingB);
                    Assert.False(
                        adjacent,
                        "two layers that both read the same tensor were placed adjacent in a run; a "
                        + "validator walking that run would compare a hand-off that does not exist.");
                }
            }
        }
    }

    [Fact]
    public void MultiInputLayerRecordsEveryInput_SoJoinsAreVisible()
    {
        // THE BLIND SPOT THIS CLOSES. Only the single-input Forward recorded; the two multi-input
        // surfaces - Forward(IReadOnlyDictionary) and Forward(params Tensor<T>[]) - did not. So
        // tracing reconstructed the straight-line stretches perfectly and silently lost every JOIN:
        // Add, Concatenate, Multiply, cross-attention, memory read/write. A tracer blind to joins is
        // not recovering a graph, it is recovering a list - and a list is exactly what tracing exists
        // to stop the validator from assuming.
        using var trace = new LayerForwardObserver<double>();

        var left = new ConvolutionalLayer<double>(outputDepth: 4, kernelSize: 3, stride: 1, padding: 1);
        var right = new ConvolutionalLayer<double>(outputDepth: 4, kernelSize: 3, stride: 1, padding: 1);
        var input = Image(3, 8, 8);

        var a = left.Forward(input);
        var b = right.Forward(input);

        // _shape directly, not Shape.ToArray(): Shape is an immutable wrapper, so ToArray()
        // materialises a fresh int[] on every call. The field is internal and AiDotNetTests is in
        // AiDotNet.Tensors' InternalsVisibleTo list, so there is no reason to pay for the copy.
        var shape = a._shape;
        // Cast disambiguates the scalar- vs vector-activation overloads, which are otherwise
        // ambiguous on a bare null.
        var add = new AddLayer<double>(new[] { shape, shape }, (IActivationFunction<double>?)null);
        var sum = add.Forward(a, b);

        Assert.NotNull(sum);

        // Two convolutions plus ONE record per Add input. Four calls, not three: the Add contributes
        // two, which is what makes it a join rather than a step.
        var addCalls = trace.Calls.Where(c => ReferenceEquals(c.Layer, add)).ToList();
        _out.WriteLine($"total recorded calls: {trace.Calls.Count}, of which AddLayer: {addCalls.Count}");

        Assert.Equal(2, addCalls.Count);
        Assert.All(addCalls, c => Assert.Same(sum, c.Output));

        // Both convolution outputs must appear as recorded inputs of the Add - by REFERENCE, not by
        // shape. Both convs emit identically-shaped tensors here precisely so a shape comparison
        // would pass while proving nothing.
        Assert.Contains(addCalls, c => ReferenceEquals(c.Input, a));
        Assert.Contains(addCalls, c => ReferenceEquals(c.Input, b));

        // And the graph must actually resolve two distinct parents for the join.
        var producers = trace.ResolveProducers();
        var addProducerIndices = trace.Calls
            .Select((c, i) => (c, i))
            .Where(x => ReferenceEquals(x.c.Layer, add))
            .Select(x => producers[x.i])
            .Where(p => p >= 0)
            .Distinct()
            .ToList();

        _out.WriteLine($"distinct producers resolved for the join: {addProducerIndices.Count}");

        Assert.Equal(2, addProducerIndices.Count);
    }
}
