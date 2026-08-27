using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Cloning;

/// <summary>
/// Clone checks that look at what a clone CONTAINS, not merely that one came back.
/// </summary>
/// <remarks>
/// The 339-layer sweep can only afford to compare ParameterCount -- materialising two parameter
/// vectors for every layer ran it past its own timeout. That leaves a gap these tests fill: a clone
/// can be non-null, the right type, and the right length while carrying the wrong values or having
/// silently dropped a child. Each case below is a defect that actually occurred.
/// </remarks>
public class CloneFidelityTests
{
    /// <summary>A clone must reproduce weights exactly, not to float precision.</summary>
    /// <remarks>
    /// Cloning a double layer was observed allocating System.Single[], which raised the question of
    /// whether weights were round-tripping through float. They are not -- but nothing asserted it,
    /// and a clone that quietly rounds every weight to seven significant digits would pass every
    /// other check in the suite: right type, right count, plausible values, slightly wrong model.
    /// </remarks>
    [Fact]
    public void Clone_PreservesPrecisionBeyondFloat()
    {
        var layer = new DenseLayer<double>(32);

        var weights = layer.GetParameters();
        for (int i = 0; i < weights.Length; i++)
        {
            // Needs far more than float's ~7 significant digits to survive.
            weights[i] = 1.0 + (i + 1) * 1e-15;
        }
        layer.SetParameters(weights);

        var clone = layer.Clone();
        var cloned = clone.GetParameters();
        var original = layer.GetParameters();

        Assert.Equal(original.Length, cloned.Length);
        for (int i = 0; i < original.Length; i++)
        {
            Assert.True(
                original[i] == cloned[i],
                $"parameter {i} changed through Clone: {original[i]:R} became {cloned[i]:R}. "
                    + "A clone that rounds weights is indistinguishable from a correct one by type "
                    + "and count alone.");
        }
    }

    /// <summary>A cloned composite must keep its child layers, not just their type name.</summary>
    /// <remarks>
    /// ParallelStreamsLayer holds its streams as IEnumerable&lt;ILayer&lt;T&gt;&gt;. That classified
    /// as a plain interface component and persisted via FormatType -- the TYPE NAME only -- so the
    /// layer cloned "successfully" with its streams gone. The clone was non-null and the right type,
    /// which is everything the sweep checked.
    /// </remarks>
    [Fact]
    public void Clone_KeepsChildLayersOfAComposite()
    {
        var streamA = new List<ILayer<double>> { new DenseLayer<double>(4) };
        var streamB = new List<ILayer<double>> { new DenseLayer<double>(4) };
        var layer = new ParallelStreamsLayer<double>(8, 4, 4, streamA, streamB);

        // Resolve the lazy shapes first: a DenseLayer built without an input width owns nothing
        // until it has seen a forward, so cloning it before that proves nothing about children.
        layer.Forward(new Tensor<double>(new[] { 1, 8 }));

        long originalCount = layer.ParameterCount;
        Assert.True(originalCount > 0, "the composite should own its children's parameters");

        var clone = layer.Clone();

        Assert.Equal(originalCount, clone.ParameterCount);
        Assert.True(
            clone.ParameterCount > 0,
            "the clone reports zero parameters, so its stream layers did not survive -- the failure "
                + "mode that persisting a collection as a bare type name produces.");
    }

    /// <summary>A layer must clone correctly while its forward activations are still resident.</summary>
    /// <remarks>
    /// The 339-layer sweep calls ResetState() between the forward probe and the clone, because
    /// holding every layer's backward activation caches at once made whichever of VAEDecoder /
    /// VAEEncoder ran first throw OutOfMemoryException -- a single 512x512 double activation is
    /// 268MB. That keeps the sweep honest about parameters but stops it exercising the case where
    /// a clone is taken while scratch is LIVE, so it is covered explicitly here.
    ///
    /// This is the state a clone is most likely to be taken in during real training (mid-epoch,
    /// activations retained for the backward pass), and it is the state ConvolutionalLayer's
    /// scratch handling was changed for: EnsureInitialized no longer pre-allocates _lastInput /
    /// _lastOutput, so a clone taken after a forward must still come back complete and independent.
    /// </remarks>
    [Theory]
    [InlineData("conv")]
    [InlineData("dense")]
    [InlineData("lstm")]
    public void CloneWithLiveActivations_IsCompleteAndIndependent(string kind)
    {
        LayerBase<double> layer;
        Tensor<double> input;

        switch (kind)
        {
            case "conv":
                layer = new ConvolutionalLayer<double>(outputDepth: 16, kernelSize: 3, stride: 1, padding: 1);
                input = Tensor<double>.CreateRandom(3, 8, 8);
                break;
            case "dense":
                layer = new DenseLayer<double>(32);
                input = Tensor<double>.CreateRandom(1, 16);
                break;
            default:
                layer = new LSTMLayer<double>(8);
                input = Tensor<double>.CreateRandom(2, 3, 4);
                break;
        }

        // Forward and deliberately DO NOT reset: the activation caches stay live across the clone.
        layer.Forward(input);

        var original = layer.GetParameters();
        Assert.NotEmpty(original.ToArray());

        var clone = (LayerBase<double>)layer.Clone();

        Assert.Equal(layer.GetType(), clone.GetType());
        Assert.Equal(layer.ParameterCount, clone.ParameterCount);
        Assert.Equal(original.ToArray(), clone.GetParameters().ToArray());

        // Independent storage: writing to the clone must not reach back into the source.
        var mutated = clone.GetParameters();
        mutated[0] = original[0] + 12345.0;
        clone.SetParameters(mutated);

        Assert.Equal(original[0], layer.GetParameters()[0]);

        // And the clone must still be usable -- a clone that inherited half-built scratch would
        // throw or return the wrong shape on its own first forward.
        var replayed = clone.Forward(input);
        Assert.NotNull(replayed);
        Assert.Equal(layer.Forward(input).Shape, replayed.Shape);
    }

}
