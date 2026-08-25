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
}
