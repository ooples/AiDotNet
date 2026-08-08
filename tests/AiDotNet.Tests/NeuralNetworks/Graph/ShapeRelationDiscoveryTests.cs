using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.NeuralNetworks.Graph;

/// <summary>
/// Checks that a layer's shape relation can be RECOVERED by running it, without any annotation.
/// </summary>
/// <remarks>
/// <para>
/// Annotation-only coverage does not scale: this codebase has ~199 layer types and 6 of them had even the
/// coarse relation kind declared. Discovery is what makes the shape system apply to all of them at once,
/// and to every layer added afterwards without anyone remembering.
/// </para>
/// <para>
/// The convolution cases are the real test. A convolution's extent relation is the one the coarse
/// mechanism could name but never compute, and the probes deliberately include sizes where the true
/// formula and "divide by stride" disagree — so a fit that reproduces every probe has genuinely recovered
/// the formula rather than a shortcut that happens to agree on round numbers.
/// </para>
/// </remarks>
public class ShapeRelationDiscoveryTests
{
    private readonly ITestOutputHelper _out;
    public ShapeRelationDiscoveryTests(ITestOutputHelper output) => _out = output;

    private static Tensor<double> Filled(int[] shape)
    {
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = ((i * 17) % 11) / 11.0;
        return t;
    }

    /// <summary>Runs a FRESH layer per probe, so a lazy layer never pins itself to the first shape.</summary>
    private static List<(int[] Input, int[] Output)> Observe(
        Func<ILayer<double>> newLayer, IEnumerable<int[]> probes)
    {
        var observations = new List<(int[], int[])>();
        foreach (var shape in probes)
        {
            try
            {
                var output = newLayer().Forward(Filled(shape)).Shape.ToArray();
                observations.Add((shape, output));
            }
            catch
            {
                // A probe the layer rejects carries no information about its relation; it is not evidence
                // of anything, so it is dropped rather than recorded as a failure to fit.
            }
        }
        return observations;
    }

    // ---------------------------------------------------------------------------------------------
    // Recovering a convolution's relation with no annotation involved
    // ---------------------------------------------------------------------------------------------

    [Theory]
    [InlineData(3, 1, 1)]   // stride 1, same padding
    [InlineData(3, 2, 1)]   // stride 2, same padding
    [InlineData(3, 2, 0)]   // stride 2, valid padding
    [InlineData(5, 3, 2)]   // larger kernel, stride 3
    [InlineData(1, 1, 0)]   // 1x1 projection
    public void Convolution_WindowRelationIsRecoveredFromTheForwardAlone(int kernel, int stride, int padding)
    {
        var probes = ShapeRelationDiscovery.ProbeShapes(new[] { 3, 24, 32 });

        var observations = Observe(
            () => new ConvolutionalLayer<double>(
                outputDepth: 6, kernelSize: kernel, stride: stride, padding: padding),
            probes);

        Assert.True(observations.Count >= 3, "not enough probes survived to fit anything");

        var inputAxes = new[] { TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width };
        var outputAxes = new[] { TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width };

        var findings = ShapeRelationDiscovery.Fit(inputAxes, outputAxes, observations);

        foreach (var (finding, i) in findings.Select((f, i) => (f, i)))
            _out.WriteLine($"  axis {outputAxes[i]}: {finding.Relation?.ToString() ?? "<none>"}  ({finding.Detail})");

        // Channels are fixed by the layer, independent of the input.
        Assert.Equal(AxisRelation.Form.Fixed, findings[0].Relation?.Kind);
        Assert.Equal(6, findings[0].Relation!.Value);

        // Height and Width follow the window formula with THIS layer's terms.
        foreach (int axis in new[] { 1, 2 })
        {
            var relation = findings[axis].Relation;
            Assert.True(relation is not null, $"axis {axis}: {findings[axis].Detail}");
            Assert.False(findings[axis].Ambiguous, findings[axis].Detail);

            // A stride-1, padding-(k-1)/2 convolution genuinely preserves extent, and Same is the correct
            // and simplest description of that - so accept either, but require the recovered relation to
            // REPRODUCE every observation, which is the property that actually matters.
            Assert.True(
                relation!.Kind is AxisRelation.Form.Window or AxisRelation.Form.Same
                    or AxisRelation.Form.Scaled,
                $"axis {axis} fitted an unexpected form: {relation.Kind}");
        }

        AssertReproducesEveryObservation(findings, inputAxes, observations);
    }

    [Fact]
    public void RecoveredRelationReproducesTheForward_ForAStrideThatDoesNotDivideEvenly()
    {
        // 24 and 32 with stride 3 and padding 0 land on inputs where "divide by stride" is wrong. If the
        // fit only matched round numbers it would break here.
        var probes = ShapeRelationDiscovery.ProbeShapes(new[] { 3, 25, 31 });
        var observations = Observe(
            () => new ConvolutionalLayer<double>(outputDepth: 4, kernelSize: 4, stride: 3, padding: 0),
            probes);

        var axes = new[] { TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width };
        var findings = ShapeRelationDiscovery.Fit(axes, axes, observations);

        AssertReproducesEveryObservation(findings, axes, observations);
    }

    // ---------------------------------------------------------------------------------------------
    // Discovery must agree with what a layer DECLARES
    // ---------------------------------------------------------------------------------------------

    [Fact]
    public void DiscoveryAgreesWithTheDeclaredContract_ForConvolution()
    {
        // Two independent routes to the same relation: one read off the declaration, one measured from
        // the forward. They must land in the same place, or one of them is lying.
        var probes = ShapeRelationDiscovery.ProbeShapes(new[] { 3, 24, 32 });
        var observations = Observe(
            () => new ConvolutionalLayer<double>(outputDepth: 6, kernelSize: 3, stride: 2, padding: 1),
            probes);

        var axes = new[] { TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width };
        var findings = ShapeRelationDiscovery.Fit(axes, axes, observations);

        var declared = new ConvolutionalLayer<double>(
            outputDepth: 6, kernelSize: 3, stride: 2, padding: 1);

        foreach (var (input, _) in observations)
        {
            var fromDeclaration = ShapeInference.InferOutputShape(declared, input);
            var fromDiscovery = Apply(findings, axes, input);

            Assert.True(
                fromDeclaration is not null && fromDiscovery is not null
                && fromDeclaration.SequenceEqual(fromDiscovery),
                $"for input [{string.Join(",", input)}] the declared contract says "
                + $"[{string.Join(",", fromDeclaration ?? Array.Empty<int>())}] but the relation measured "
                + $"from the forward says [{string.Join(",", fromDiscovery ?? Array.Empty<int>())}].");
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Probe quality is part of the contract
    // ---------------------------------------------------------------------------------------------

    [Fact]
    public void ProbeShapes_VaryEachAxisIndependently()
    {
        // If two axes ever move together, Same(A) and Same(B) become indistinguishable and the fit is
        // right on the probes and wrong on the first input that separates them.
        var probes = ShapeRelationDiscovery.ProbeShapes(new[] { 3, 16, 16 });
        var baseShape = probes[0];

        for (int axis = 0; axis < baseShape.Length; axis++)
        {
            var changed = probes.Skip(1)
                .Where(p => p[axis] != baseShape[axis])
                .ToList();
            Assert.True(changed.Count > 0, $"no probe varies axis {axis}, so it cannot be identified");

            foreach (var probe in changed)
            {
                int differing = Enumerable.Range(0, baseShape.Length).Count(i => probe[i] != baseShape[i]);
                Assert.Equal(1, differing);
            }
        }
    }

    [Fact]
    public void EqualAxesAreReportedAmbiguous_NotGuessed()
    {
        // Every probe has Height == Width, so Same(Height) and Same(Width) both explain the data. The
        // honest answer is "the probes do not separate these", not a coin flip.
        var observations = new List<(int[], int[])>
        {
            (new[] { 8, 8 }, new[] { 8 }),
            (new[] { 5, 5 }, new[] { 5 }),
        };

        var findings = ShapeRelationDiscovery.Fit(
            new[] { TensorAxis.Height, TensorAxis.Width },
            new[] { TensorAxis.Height },
            observations);

        Assert.True(findings[0].Ambiguous);
        Assert.Contains("ambiguous", findings[0].Detail, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ConstantOutputFromASingleProbe_IsNotCalledFixed()
    {
        // One observation makes every axis look constant. That is a fact about the probe set, not the
        // layer, and reporting Fixed would be an unearned claim.
        var findings = ShapeRelationDiscovery.Fit(
            new[] { TensorAxis.Channels },
            new[] { TensorAxis.Channels },
            new List<(int[], int[])> { (new[] { 7 }, new[] { 4 }) });

        Assert.NotEqual(AxisRelation.Form.Fixed, findings[0].Relation?.Kind);
    }

    [Fact]
    public void UnexplainableAxis_ReportsNoRelation_WithAReason()
    {
        // Output sizes that follow no Same/Fixed/Scaled/Window rule at all.
        var observations = new List<(int[], int[])>
        {
            (new[] { 4 }, new[] { 3 }),
            (new[] { 5 }, new[] { 11 }),
            (new[] { 6 }, new[] { 2 }),
        };

        var findings = ShapeRelationDiscovery.Fit(
            new[] { TensorAxis.Features }, new[] { TensorAxis.Features }, observations);

        Assert.Null(findings[0].Relation);
        Assert.False(string.IsNullOrWhiteSpace(findings[0].Detail));
    }

    // ---------------------------------------------------------------------------------------------

    private static int[]? Apply(
        IReadOnlyList<ShapeRelationDiscovery.AxisFinding> findings,
        IReadOnlyList<TensorAxis> inputAxes,
        IReadOnlyList<int> inputShape)
    {
        var named = new Dictionary<TensorAxis, int>();
        for (int i = 0; i < inputAxes.Count && i < inputShape.Count; i++) named[inputAxes[i]] = inputShape[i];

        var result = new int[findings.Count];
        for (int i = 0; i < findings.Count; i++)
        {
            if (findings[i].Relation is null) return null;
            if (!findings[i].Relation!.TryResolve(named, out result[i])) return null;
        }
        return result;
    }

    private static void AssertReproducesEveryObservation(
        IReadOnlyList<ShapeRelationDiscovery.AxisFinding> findings,
        IReadOnlyList<TensorAxis> axes,
        IReadOnlyList<(int[] Input, int[] Output)> observations)
    {
        foreach (var (input, expected) in observations)
        {
            var predicted = Apply(findings, axes, input);
            Assert.True(
                predicted is not null && predicted.SequenceEqual(expected),
                $"the fitted relation predicts [{string.Join(",", predicted ?? Array.Empty<int>())}] for "
                + $"input [{string.Join(",", input)}] but the layer produced "
                + $"[{string.Join(",", expected)}]. A fit that does not reproduce its own observations "
                + "has not recovered anything.");
        }
    }
}
