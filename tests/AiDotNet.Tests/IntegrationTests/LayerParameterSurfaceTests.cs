using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests;

/// <summary>
/// Asserts, for every constructable layer, that <c>ParameterCount</c> equals
/// <c>GetParameters().Length</c>.
/// </summary>
/// <remarks>
/// <para>
/// This is the invariant the MODEL-level failures are downstream of.
/// <c>NeuralNetworkBase.ParameterCount</c> sums <c>layer.ParameterCount</c> while
/// <c>NeuralNetworkBase.GetParameters()</c> sums <c>layer.GetParameters().Length</c> — the same
/// list, but a DIFFERENT question asked of each layer. Wherever one layer answers those two
/// questions differently, every model containing it reports a mismatch, and the model looks like
/// the culprit. Three of the models failing CI (MusicSourceSeparator, RoomImpulseResponse,
/// OpenVoiceV2) override neither member, which is the proof: the divergence is underneath them.
/// </para>
/// <para>
/// PyTorch cannot express this bug. <c>nn.Module.parameters()</c> is the sole registry, populated
/// automatically by <c>__setattr__</c> interception, and the count is
/// <c>sum(p.numel() for p in model.parameters())</c> — a fold over the very tensors the iterator
/// yields. There is no per-module count to drift from it. The equivalent guarantee here is that
/// each layer's two surfaces describe one set of tensors; once that holds, a model summing either
/// one gets the same answer, and models stop needing to hand-roll agreement between them.
/// </para>
/// <para>
/// Reports every violation in one message rather than failing at the first, for the same reason
/// the model sweep does: fixing one and re-running tells you nothing about how many remain.
/// </para>
/// </remarks>
[Collection("ParameterSweeps")]
[Trait("Category", "Sweep")]
public class LayerParameterSurfaceTests
{
    private readonly ITestOutputHelper _output;

    public LayerParameterSurfaceTests(ITestOutputHelper output) => _output = output;

    [Fact(Timeout = 900000)]
    public async System.Threading.Tasks.Task AllLayers_ParameterCountMatchesGetParameters()
    {
        await System.Threading.Tasks.Task.Yield();

        var violations = new List<string>();
        int checkedCount = 0, unconstructable = 0, unsized = 0;

        var logPath = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "layer-parameter-surface.txt");
        System.IO.StreamWriter? log = null;
        try { log = new System.IO.StreamWriter(logPath, append: false) { AutoFlush = true }; } catch { }

        foreach (var closed in GetConstructableLayerTypes())
        {
            var name = closed.FullName ?? closed.Name;
            log?.WriteLine($"[measuring] {name}");

            object? layer;
            try { layer = TryConstruct(closed); }
            catch { layer = null; }
            if (layer is null) { unconstructable++; continue; }

            try
            {
                long declared = Convert.ToInt64(
                    closed.GetProperty("ParameterCount")!.GetValue(layer));

                // A layer whose shape is still deferred legitimately reports nothing from BOTH
                // surfaces; that is agreement, not a violation, and it is the state the deferred
                // layers were fixed INTO. Counted so the coverage is honest rather than inflated.
                var uninitProp = closed.GetProperty("HasUninitializedParameters",
                    BindingFlags.Public | BindingFlags.Instance | BindingFlags.FlattenHierarchy);
                bool pending = uninitProp?.GetValue(layer) is true;

                var m = closed.GetMethod("GetParameters", BindingFlags.Public | BindingFlags.Instance,
                    null, Type.EmptyTypes, null);
                if (m is null) continue;
                var vec = m.Invoke(layer, null);
                long actual = vec is null ? 0
                    : Convert.ToInt64(vec.GetType().GetProperty("Length")!.GetValue(vec));

                checkedCount++;
                if (declared == actual) { if (pending) unsized++; continue; }

                var row = $"{name}: ParameterCount={declared}, GetParameters().Length={actual} " +
                          $"(difference {declared - actual}){(pending ? " [deferred]" : "")}";
                violations.Add(row);
                log?.WriteLine("VIOLATION " + row);
            }
            catch (Exception ex)
            {
                unconstructable++;
                _output.WriteLine($"UNMEASURABLE {name}: {ex.GetBaseException().GetType().Name}");
            }
        }

        log?.Dispose();

        _output.WriteLine($"Checked {checkedCount} layers; {unsized} agree at zero (deferred); " +
                          $"{unconstructable} not constructable; {violations.Count} violations.");
        foreach (var v in violations.OrderBy(v => v, StringComparer.Ordinal))
            _output.WriteLine("  " + v);

        Assert.True(violations.Count == 0,
            $"{violations.Count} layer(s) report a ParameterCount that disagrees with " +
            "GetParameters().Length.\n\n" +
            "Every model containing such a layer reports a mismatch of its own, because " +
            "NeuralNetworkBase.ParameterCount sums layer.ParameterCount while GetParameters() sums " +
            "layer.GetParameters().Length. Fixing the layer fixes every model that holds it; " +
            "fixing the models one at a time does not.\n\n" +
            string.Join("\n", violations.OrderBy(v => v, StringComparer.Ordinal).Select(v => "  " + v)));
    }

    private static object? TryConstruct(Type closed)
    {
        var ctor = closed.GetConstructors(BindingFlags.Public | BindingFlags.Instance)
            .Where(c => c.GetParameters().All(p => p.HasDefaultValue) || c.GetParameters().Length == 0)
            .OrderBy(c => c.GetParameters().Length)
            .FirstOrDefault();
        if (ctor is null) return null;
        var args = ctor.GetParameters()
            .Select(p => p.DefaultValue == DBNull.Value ? null : p.DefaultValue).ToArray();
        return ctor.Invoke(args);
    }

    private static IEnumerable<Type> GetConstructableLayerTypes()
    {
        var assembly = typeof(AiDotNet.Models.ModelMetadata<>).Assembly;
        foreach (var open in assembly.GetTypes()
                     .Where(t => t.IsClass && !t.IsAbstract && t.IsGenericTypeDefinition)
                     .Where(t => t.GetGenericArguments().Length == 1))
        {
            Type closed;
            try { closed = open.MakeGenericType(typeof(double)); } catch { continue; }

            // LayerBase<T> descendants only — this is about the layer contract, not models.
            bool isLayer = false;
            for (var b = closed.BaseType; b is not null; b = b.BaseType)
                if (b.IsGenericType && b.GetGenericTypeDefinition().Name.StartsWith("LayerBase", StringComparison.Ordinal))
                { isLayer = true; break; }
            if (!isLayer) continue;

            if (!closed.GetConstructors(BindingFlags.Public | BindingFlags.Instance)
                    .Any(c => c.GetParameters().Length == 0 || c.GetParameters().All(p => p.HasDefaultValue)))
                continue;

            yield return closed;
        }
    }
}
