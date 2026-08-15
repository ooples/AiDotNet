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

}
