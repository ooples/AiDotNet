using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.NeuralNetworks.Graph;

/// <summary>
/// A model's own declared input layout must share a rank with its first layer's.
/// </summary>
/// <remarks>
/// <para>
/// BOTH SIDES ARE DECLARATIONS, WHICH IS WHAT MAKES THIS SOUND. An earlier attempt compared the
/// ARCHITECTURE's input shape against the first layer and produced 179 false positives, because the
/// architecture is supplied by the caller: the harness handed every model a 3-D image and then
/// reported TTS models for opening with an EmbeddingLayer that takes rank 1-2. That check was
/// measuring its own input. This one reads <c>[TensorLayout(Direction = Input)]</c> off the MODEL
/// type and off the LAYER type, so neither operand comes from the test.
/// </para>
/// <para>
/// It closes the boundary nothing else looks at.
/// <c>NeuralNetworkBase.ReportLayerContractMismatches</c> validates layer-to-layer ADJACENCY - it
/// returns early when <c>Layers.Count &lt; 2</c>, so it only inspects pairs INSIDE the stack. The
/// conformance sweep checks the far end, the model's output contract against what Predict returns.
/// Neither looks at where the declared input meets the first layer, and
/// <c>TensorLayoutAttribute.AcceptsRank</c> was never called from the model side at all.
/// </para>
/// <para>
/// Models that declare no input layout are COUNTED, not failed - that count is the backlog, the same
/// shape of ladder ADNSHAPE006 climbed from 85 to 0. A model that declares one and contradicts its
/// own first layer is a defect, because Predict cannot succeed on the input the model advertises.
/// </para>
/// </remarks>
public class ModelInputMeetsFirstLayerTests
{
    private readonly ITestOutputHelper _out;
    public ModelInputMeetsFirstLayerTests(ITestOutputHelper output) => _out = output;

    private const int Extent = 8;

    [Fact]
    public void AModelsDeclaredInputRankIsOneItsFirstLayerAccepts()
    {
        var models = typeof(NeuralNetworkBase<>).Assembly.GetTypes()
            .Where(t => t.IsClass && !t.IsAbstract && t.IsGenericTypeDefinition
                        && t.GetGenericArguments().Length == 1
                        && DerivesFromNeuralNetworkBase(t))
            .OrderBy(t => t.Name, StringComparer.Ordinal)
            .ToList();

        int checkedCount = 0, modelUndeclared = 0, layerUndeclared = 0;
        var mismatched = new List<string>();
        var skipped = new List<string>();

        foreach (var open in models)
        {
            Type closed;
            try { closed = open.MakeGenericType(typeof(double)); }
            catch { continue; }

            // The model's OWN declaration, read from the open generic type so no instance is needed
            // for this half - inherited attributes included, since a family base declares for all.
            var modelRanks = AcceptedRanks(open);
            if (modelRanks.Count == 0) { modelUndeclared++; continue; }

            object? model = null;
            try
            {
                model = Construct(closed);
                if (model is null) { skipped.Add($"{open.Name}: no architecture constructor"); continue; }

                var first = FirstLayer(model);
                if (first is null) { skipped.Add($"{open.Name}: no layers"); continue; }

                var layerRanks = AcceptedRanks(first.GetType());
                if (layerRanks.Count == 0) { layerUndeclared++; continue; }

                checkedCount++;
                if (modelRanks.Overlaps(layerRanks)) continue;

                mismatched.Add(
                    $"{open.Name}: declares input rank(s) [{Fmt(modelRanks)}] but its first layer "
                    + $"{first.GetType().Name} accepts [{Fmt(layerRanks)}]");
            }
            catch (Exception ex) { skipped.Add($"{open.Name}: {Unwrap(ex).GetType().Name}"); }
            finally { (model as IDisposable)?.Dispose(); }
        }

        _out.WriteLine($"models checked                        : {checkedCount}");
        _out.WriteLine($"model declares no input layout (backlog): {modelUndeclared}");
        _out.WriteLine($"first layer declares no input layout  : {layerUndeclared}");
        _out.WriteLine($"MISMATCHED                            : {mismatched.Count}");
        _out.WriteLine($"skipped                               : {skipped.Count}");
        foreach (var m in mismatched) _out.WriteLine($"  MISMATCH: {m}");

        // Assert the EXERCISED count. A run where nothing constructed would otherwise pass while
        // proving nothing - the vacuous-sweep failure mode that hid 13 dead [ElementWiseShape]
        // contracts, and that a previous version of this very check tripped over (0 checked, 907
        // "no input shape", caught only because the guard was here).
        Assert.True(checkedCount > 0,
            "no model was checked, so this proves nothing about the declared-input boundary");

        // COUNTED, NOT ASSERTED - YET. First run: 741 checked, 27 mismatched. Failing on 27 would
        // redden the branch against work that is unfinished rather than wrong, which is the exact
        // reason ADNSHAPE006 entered as a suppressed warning at 85 of ~270 layers and was promoted
        // to Error only when it reached zero. Same ladder here: the number is printed on every run
        // so the backlog is readable, and REPLACING THIS BLOCK WITH AN Assert IS THE PROMOTION.
        //
        // Each of the 27 needs a judgement that cannot be made in bulk - whether the MODEL's
        // declared rank is wrong or its FIRST LAYER's is - so they are worked individually, not
        // swept. What must not happen is the count drifting up unnoticed, and printing it every run
        // is what prevents that.
        Assert.True(mismatched.Count <= 27,
            $"the declared-input mismatch backlog GREW to {mismatched.Count} (was 27). A new model "
            + "declares an input rank its own first layer rejects, so Predict cannot succeed on the "
            + "input that model advertises."
            + Environment.NewLine + string.Join(Environment.NewLine, mismatched));
    }

    /// <summary>Ranks a type's own <c>[TensorLayout(Input)]</c> declarations accept.</summary>
    private static HashSet<int> AcceptedRanks(Type type)
    {
        var ranks = new HashSet<int>();
        foreach (var l in type.GetCustomAttributes(typeof(TensorLayoutAttribute), inherit: true)
                              .Cast<TensorLayoutAttribute>()
                              .Where(a => a.Direction == TensorLayoutDirection.Input
                                          && a.Axes is { Length: > 0 }))
        {
            // AcceptsRank is the authority - it already encodes BatchOptional, so a
            // [Batch?, Channels, Height, Width] layout accepts both 3 and 4 and neither is
            // reconstructable from Axes.Length alone.
            for (int r = 1; r <= 6; r++)
                if (l.AcceptsRank(r)) ranks.Add(r);
        }
        return ranks;
    }

    private static string Fmt(IEnumerable<int> ranks) => string.Join(", ", ranks.OrderBy(r => r));

    private static bool DerivesFromNeuralNetworkBase(Type type)
    {
        for (var t = type.BaseType; t is not null; t = t.BaseType)
        {
            if (t.IsGenericType && t.GetGenericTypeDefinition() == typeof(NeuralNetworkBase<>)) return true;
        }
        return false;
    }

    private static LayerBase<double>? FirstLayer(object model)
    {
        try
        {
            var layers = (model as NeuralNetworkBase<double>)?.Layers;
            if (layers is null) return null;
            foreach (var l in layers) if (l is LayerBase<double> lb) return lb;
            return null;
        }
        catch { return null; }
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
            inputDepth: 3, inputHeight: Extent, inputWidth: Extent, outputSize: 4);

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
