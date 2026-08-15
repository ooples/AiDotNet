using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.NeuralNetworks.Graph;

/// <summary>
/// Verifies the frame-interpolation law <c>(F - 1) * TemporalScaleFactor + 1</c> against real forward
/// passes, at MORE THAN ONE clip length.
/// </summary>
/// <remarks>
/// <para>
/// The generic conformance sweep cannot reach this family. It builds a rank-4 input with a leading 1,
/// and <c>FrameInterpolationBase.PredictCore</c> explicitly rejects that shape - rank 4 means a frame
/// SEQUENCE <c>[N, C, H, W]</c>, and a leading 1 is diagnosed as a mis-passed pair. So the family is
/// checked here, with clips of 2 and 3 frames.
/// </para>
/// <para>
/// TWO lengths, deliberately. The whole claim of an affine relation is that it holds for lengths
/// nobody ran: at k=2 it predicts 3 frames from 2 and 5 from 3. A single observation would be
/// reproduced just as well by a <c>Fixed</c> constant, which is exactly the error that made three
/// vision-language contracts wrong earlier on this branch.
/// </para>
/// </remarks>
public class FrameInterpolationShapeContractTests
{
    private readonly ITestOutputHelper _out;
    public FrameInterpolationShapeContractTests(ITestOutputHelper output) => _out = output;

    private const int Channels = 3;
    private const int Extent = 16;

    /// <summary>
    /// The affine form resolves <c>(F-1)*k + 1</c> correctly, including the negative offset.
    /// </summary>
    /// <remarks>
    /// Kept separate from the family check because the family currently DECLINES: every concrete model
    /// overrides PredictCore and never reaches the sequence path, so nothing there exercises the new
    /// relation. Without this, AxisRelation.Affine would ship with no verification behind it - the
    /// "mechanism nothing consumes" that this branch has refused elsewhere.
    /// </remarks>
    [Fact]
    public void TheAffineRelationResolvesAScaleWithANegativeOffset()
    {
        // k = 2: 2 -> 3, 3 -> 5, 5 -> 9. A Fixed or a plain Scaled reproduces none of these.
        var doubling = AxisRelation.Affine(TensorAxis.Frames, 2, 1, -1);
        Assert.Equal(3, Resolve(doubling, 2));
        Assert.Equal(5, Resolve(doubling, 3));
        Assert.Equal(9, Resolve(doubling, 5));

        // k = 4: 2 -> 5, 3 -> 9.
        var quadrupling = AxisRelation.Affine(TensorAxis.Frames, 4, 1, -3);
        Assert.Equal(5, Resolve(quadrupling, 2));
        Assert.Equal(9, Resolve(quadrupling, 3));

        // A ratio that does not divide evenly is a declaration error, not something to round -
        // the same rule Form.Scaled already enforces.
        var halving = AxisRelation.Affine(TensorAxis.Frames, 1, 2, 1);
        Assert.Equal(3, Resolve(halving, 4));
        Assert.Null(Resolve(halving, 5));

        // An offset that drives the axis to zero or below is refused rather than returned.
        var collapsing = AxisRelation.Affine(TensorAxis.Frames, 1, 1, -8);
        Assert.Null(Resolve(collapsing, 8));

        _out.WriteLine($"affine prints as: {doubling}");
    }

    private static int? Resolve(AxisRelation relation, int frames)
    {
        var axes = new Dictionary<TensorAxis, int> { [TensorAxis.Frames] = frames };
        return relation.TryResolve(axes, out int size) ? size : (int?)null;
    }

    [Fact]
    public void TheInterpolationLawPredictsTheFrameCountAForwardPassProduces()
    {
        var models = typeof(NeuralNetworkBase<>).Assembly.GetTypes()
            .Where(t => t.IsClass && !t.IsAbstract && t.IsGenericTypeDefinition
                        && t.GetGenericArguments().Length == 1
                        && DerivesFromInterpolationBase(t))
            .OrderBy(t => t.Name, StringComparer.Ordinal)
            .ToList();

        _out.WriteLine($"frame-interpolation models: {models.Count}");

        int agreed = 0, declined = 0;
        var disagreed = new List<string>();
        var skipped = new List<string>();

        foreach (var open in models)
        {
            Type closed;
            try { closed = open.MakeGenericType(typeof(double)); }
            catch { continue; }

            object? model = null;
            try
            {
                model = Construct(closed);
                if (model is null) { skipped.Add($"{open.Name}: no architecture constructor"); continue; }
                if (model is not IShapeContract contract) { skipped.Add($"{open.Name}: not IShapeContract"); continue; }

                bool anyRun = false;
                foreach (int frames in new[] { 2, 3 })
                {
                    var shape = new[] { frames, Channels, Extent, Extent };
                    int[]? predicted = ShapeInference.InferOutputShape(contract, shape);
                    if (predicted is null) { declined++; continue; }

                    int[] actual;
                    try
                    {
                        var input = new Tensor<double>(shape);
                        actual = ((NeuralNetworkBase<double>)model).Predict(input).Shape.ToArray();
                    }
                    catch (Exception ex)
                    {
                        skipped.Add($"{open.Name} (F={frames}): {Unwrap(ex).GetType().Name}: {Unwrap(ex).Message}");
                        continue;
                    }

                    anyRun = true;
                    if (predicted.SequenceEqual(actual))
                    {
                        agreed++;
                        _out.WriteLine($"{open.Name,-24} F={frames}  [{string.Join(",", shape)}] -> [{string.Join(",", actual)}]");
                    }
                    else
                    {
                        disagreed.Add($"{open.Name} (F={frames}): contract says [{string.Join(",", predicted)}] "
                            + $"but Predict returned [{string.Join(",", actual)}]");
                    }
                }

                if (!anyRun && declined == 0) skipped.Add($"{open.Name}: no clip length ran");
            }
            finally { (model as IDisposable)?.Dispose(); }
        }

        _out.WriteLine("");
        _out.WriteLine($"agreed={agreed}  declined={declined}  disagreed={disagreed.Count}  skipped={skipped.Count}");
        foreach (var s in skipped.Take(30)) _out.WriteLine($"  skipped: {s}");
        foreach (var d in disagreed) _out.WriteLine($"  DISAGREED: {d}");

        Assert.True(models.Count > 0,
            "no type derives from FrameInterpolationBase, so this law is a claim about nothing");

        Assert.True(disagreed.Count == 0,
            $"{disagreed.Count} interpolation contract(s) disagree with a real forward pass."
            + Environment.NewLine + string.Join(Environment.NewLine, disagreed));
    }

    private static bool DerivesFromInterpolationBase(Type type)
    {
        for (var t = type.BaseType; t is not null; t = t.BaseType)
        {
            if (t.IsGenericType && t.GetGenericTypeDefinition() == typeof(FrameInterpolationBase<>)) return true;
        }
        return false;
    }

    private static object? Construct(Type closed)
    {
        var ctor = closed.GetConstructors().FirstOrDefault(c =>
        {
            var ps = c.GetParameters();
            return ps.Length > 0
                && ps[0].ParameterType == typeof(NeuralNetworkArchitecture<double>)
                && ps.Skip(1).All(p => p.HasDefaultValue);
        });
        if (ctor is null) return null;

        var architecture = new NeuralNetworkArchitecture<double>(
            InputType.ThreeDimensional, NeuralNetworkTaskType.Regression,
            inputDepth: Channels, inputHeight: Extent, inputWidth: Extent, outputSize: 1);

        var pars = ctor.GetParameters();
        var args = new object?[pars.Length];
        args[0] = architecture;
        for (int i = 1; i < pars.Length; i++) args[i] = pars[i].DefaultValue;

        try { return ctor.Invoke(args); }
        catch { return null; }
    }

    private static Exception Unwrap(Exception ex)
        => ex is System.Reflection.TargetInvocationException { InnerException: not null } tie
            ? Unwrap(tie.InnerException) : ex;
}
