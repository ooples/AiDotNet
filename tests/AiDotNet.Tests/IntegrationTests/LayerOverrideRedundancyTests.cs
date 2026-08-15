using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests;

/// <summary>
/// Reports which hand-written <c>GetParameters()</c> overrides are PROVABLY redundant — that is,
/// value-for-value identical to what <c>LayerBase</c> now produces — and are therefore safe to
/// delete.
/// </summary>
/// <remarks>
/// <para>
/// Deleting an override is not safe merely because the two agree on LENGTH. The base emits
/// <c>Parameters</c>, then this layer's registered tensors, then each sub-layer, in REGISTRATION
/// order; a hand-written getter concatenates in whatever order its author wrote the fields. If
/// those orders differ the totals still match while the layout silently changes, which invalidates
/// every saved checkpoint and mis-pairs <c>SetParameters</c> — a strictly worse failure than the
/// count mismatch this work set out to fix, and one no count-based gate can see.
/// </para>
/// <para>
/// So this compares the actual VALUES elementwise. A layer is reported redundant only when the
/// override and the base-order reconstruction agree at every index, which makes deletion a
/// no-op by construction rather than by inspection.
/// </para>
/// <para>
/// Reported, never enforced: an override that legitimately differs (weight tying that shares one
/// tensor across two slots, a GAN reading frozen modules) is a correct override, not a defect. The
/// output is a work-list for deletion, not a contract.
/// </para>
/// </remarks>
[Collection("ParameterSweeps")]
[Trait("Category", "Sweep")]
public class LayerOverrideRedundancyTests
{
    private readonly ITestOutputHelper _output;

    public LayerOverrideRedundancyTests(ITestOutputHelper output) => _output = output;

    [Fact(Timeout = 900000)]
    public async System.Threading.Tasks.Task Report_WhichGetParametersOverridesAreRedundant()
    {
        await System.Threading.Tasks.Task.Yield();

        var redundant = new List<string>();
        var differing = new List<string>();
        int skipped = 0;
        var unresolved = new List<string>();

        var logPath = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "override-redundancy.txt");
        System.IO.StreamWriter? log = null;
        try { log = new System.IO.StreamWriter(logPath, append: false) { AutoFlush = true }; } catch { }

        foreach (var closed in GetConstructableLayerTypes())
        {
            var name = closed.FullName ?? closed.Name;

            // Only layers that actually declare the override are candidates.
            var gp = closed.GetMethod("GetParameters", BindingFlags.Public | BindingFlags.Instance,
                null, Type.EmptyTypes, null);
            if (gp is null || gp.DeclaringType == typeof(object)) { skipped++; continue; }
            if (gp.DeclaringType is null || !gp.DeclaringType.Name.StartsWith(closed.Name.Split('`')[0], StringComparison.Ordinal))
            { skipped++; continue; }

            object? layer;
            try { layer = TryConstruct(closed); } catch { layer = null; }
            if (layer is null) { skipped++; continue; }

            try
            {
                var actual = gp.Invoke(layer, null);
                if (actual is null) { skipped++; continue; }
                int len = Convert.ToInt32(actual.GetType().GetProperty("Length")!.GetValue(actual));

                var expected = BaseOrderValues(layer, closed);
                if (expected is null) { skipped++; continue; }

                bool same = expected.Count == len;
                if (same)
                {
                    var idx = actual.GetType().GetProperty("Item", new[] { typeof(int) });
                    for (int i = 0; i < len && same; i++)
                    {
                        var v = Convert.ToDouble(idx!.GetValue(actual, new object[] { i }));
                        if (Math.Abs(v - expected[i]) > 1e-12) same = false;
                    }
                }

                // Equality at ZERO proves nothing. An unresolved layer returns an empty vector from
                // both the override and the base reconstruction, which says only that neither has
                // been sized yet -- not that the override is redundant. Counting those as
                // "safe to delete" is how a report of 29 candidates turned out to contain 3 real
                // ones, and acting on it would have deleted 26 overrides on no evidence at all.
                if (same && len == 0)
                {
                    unresolved.Add(name);
                    log?.WriteLine("UNRESOLVED (no evidence either way) " + name);
                }
                else if (same)
                {
                    redundant.Add($"{name} ({len} values)");
                    log?.WriteLine("REDUNDANT " + name);
                }
                else
                {
                    differing.Add($"{name}: override={len}, base-order={expected.Count}");
                    log?.WriteLine("DIFFERS " + name);
                }
            }
            catch { skipped++; }
        }

        log?.Dispose();

        _output.WriteLine($"REDUNDANT (safe to delete): {redundant.Count}");
        foreach (var r in redundant.OrderBy(x => x, StringComparer.Ordinal)) _output.WriteLine("  " + r);
        _output.WriteLine($"DIFFERS (keep, or migrate deliberately): {differing.Count}");
        foreach (var d in differing.OrderBy(x => x, StringComparer.Ordinal)) _output.WriteLine("  " + d);
        _output.WriteLine($"UNRESOLVED (empty both sides, proves nothing): {unresolved.Count}");
        _output.WriteLine($"skipped (not constructable): {skipped}");

        Assert.True(true);
    }

    /// <summary>Reproduces LayerBase.GetParameters' order exactly: Parameters, own tensors, sub-layers.</summary>
    private static List<double>? BaseOrderValues(object layer, Type t)
    {
        var values = new List<double>();

        var pField = t.GetProperty("Parameters", BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.FlattenHierarchy)
                     ?? (MemberInfo?)t.GetField("Parameters", BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.FlattenHierarchy) as PropertyInfo;
        var pVal = pField is PropertyInfo pi ? pi.GetValue(layer)
                 : t.GetField("Parameters", BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.FlattenHierarchy)?.GetValue(layer);
        if (pVal is not null) AppendVector(pVal, values);

        if (t.GetMethod("GetTrainableParameters", BindingFlags.Public | BindingFlags.Instance)
              ?.Invoke(layer, null) is IEnumerable tensors)
            foreach (var ten in tensors) { if (ten is not null) AppendTensor(ten, values); }

        if (t.GetMethod("GetSubLayers", BindingFlags.Public | BindingFlags.Instance)
              ?.Invoke(layer, null) is IEnumerable subs)
            foreach (var s in subs)
            {
                if (s is null) continue;
                var sv = s.GetType().GetMethod("GetParameters", BindingFlags.Public | BindingFlags.Instance,
                    null, Type.EmptyTypes, null)?.Invoke(s, null);
                if (sv is not null) AppendVector(sv, values);
            }

        return values;
    }

    private static void AppendVector(object vec, List<double> into)
    {
        var lenProp = vec.GetType().GetProperty("Length");
        if (lenProp is null) return;
        int n = Convert.ToInt32(lenProp.GetValue(vec));
        var idx = vec.GetType().GetProperty("Item", new[] { typeof(int) });
        if (idx is null) return;
        for (int i = 0; i < n; i++) into.Add(Convert.ToDouble(idx.GetValue(vec, new object[] { i })));
    }

    private static void AppendTensor(object ten, List<double> into)
    {
        var lenProp = ten.GetType().GetProperty("Length");
        var getFlat = ten.GetType().GetMethod("GetFlat", new[] { typeof(int) });
        if (lenProp is null || getFlat is null) return;
        int n = Convert.ToInt32(lenProp.GetValue(ten));
        for (int i = 0; i < n; i++) into.Add(Convert.ToDouble(getFlat.Invoke(ten, new object[] { i })));
    }

    private static object? TryConstruct(Type closed)
    {
        var ctor = closed.GetConstructors(BindingFlags.Public | BindingFlags.Instance)
            .Where(c => c.GetParameters().Length == 0 || c.GetParameters().All(p => p.HasDefaultValue))
            .OrderBy(c => c.GetParameters().Length).FirstOrDefault();
        if (ctor is null) return null;
        return ctor.Invoke(ctor.GetParameters()
            .Select(p => p.DefaultValue == DBNull.Value ? null : p.DefaultValue).ToArray());
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
            bool isLayer = false;
            for (var b = closed.BaseType; b is not null; b = b.BaseType)
                if (b.IsGenericType && b.GetGenericTypeDefinition().Name.StartsWith("LayerBase", StringComparison.Ordinal))
                { isLayer = true; break; }
            if (!isLayer) continue;
            yield return closed;
        }
    }
}
