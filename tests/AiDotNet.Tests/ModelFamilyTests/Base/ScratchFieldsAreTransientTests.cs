using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Every field marked <c>[Scratch]</c> must actually be transient: a forward pass has to rewrite it.
/// </summary>
/// <remarks>
/// <para>
/// <c>[Scratch]</c> is the escape hatch that removes a tensor field from the parameter surface, and
/// it is the one classification whose mistakes are invisible. A weight marked <c>[Scratch]</c>
/// vanishes from <see cref="AiDotNet.NeuralNetworks.NeuralNetworkBase{T}.ParameterCount"/> AND from
/// <c>GetParameters</c>, so the count-vs-vector contract test still passes — the two agree on a
/// wrong answer. The weight is simply never trained, never saved, and nothing says so.
/// </para>
/// <para>
/// The distinguishing property is mechanical rather than a matter of judgement. Genuine scratch is
/// RECOMPUTED: run the model twice on different inputs and its contents change, because the forward
/// pass writes it. A weight does not — it holds whatever initialization or training put there.
/// So: snapshot every <c>[Scratch]</c> field, run a forward pass on one input, run another on a
/// different input, and require the field to have been written. A field that never changes across
/// two different inputs is not scratch, and this fails with its name.
/// </para>
/// <para>
/// The converse is not tested here and cannot be: a field that legitimately holds the same value for
/// two inputs (a mask that happens to match, a cache keyed on a shape both inputs share) would look
/// static. That is why the assertion is scoped to fields the author explicitly marked — the claim
/// being checked is theirs, and this only asks them to be right about it.
/// </para>
/// </remarks>
public class ScratchFieldsAreTransientTests
{
    /// <summary>Marker names, matched by simple name so the attribute can live anywhere.</summary>
    private static bool IsScratch(FieldInfo f) =>
        f.GetCustomAttributes().Any(a => a.GetType().Name is "ScratchAttribute" or "Scratch");

    private static IEnumerable<FieldInfo> ScratchFieldsOf(object model) =>
        model.GetType()
             .GetFields(BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public)
             .Where(IsScratch);

    /// <summary>
    /// A cheap structural fingerprint of a tensor field's contents. Null when the field is unset,
    /// which is itself a legitimate scratch state before the first forward.
    /// </summary>
    private static string? Fingerprint(object? value)
    {
        switch (value)
        {
            case null:
                return null;
            case Tensor<double> td:
                return Hash(td.Length, i => td[i]);
            case Tensor<float> tf:
                return Hash(tf.Length, i => tf[i]);
            case Vector<double> vd:
                return Hash(vd.Length, i => vd[i]);
            case Vector<float> vf:
                return Hash(vf.Length, i => vf[i]);
            default:
                return null;   // not a shape this check understands; skipped rather than guessed at
        }
    }

    private static string Hash(int length, Func<int, double> at)
    {
        // Sample rather than read everything: these run across the whole model family and a
        // foundation-scale tensor would dominate the suite. Sampling is enough to notice a rewrite.
        unchecked
        {
            double acc = length;
            int step = Math.Max(1, length / 64);
            for (int i = 0; i < length; i += step) acc = acc * 31.0 + at(i);
            return acc.ToString("R");
        }
    }

    /// <summary>
    /// Asserts the claim for one already-constructed model, given two forward passes the caller
    /// drives. Model-family test bases call this; it is deliberately not a [Fact] of its own,
    /// because constructing every model here would duplicate the fixtures those bases already own.
    /// </summary>
    public static void AssertScratchFieldsAreRewritten(object model, Action runForwardA, Action runForwardB)
    {
        var fields = ScratchFieldsOf(model).ToList();
        if (fields.Count == 0) return;

        runForwardA();
        var afterA = fields.ToDictionary(f => f, f => Fingerprint(f.GetValue(model)));

        runForwardB();
        var stale = new List<string>();
        foreach (var f in fields)
        {
            var before = afterA[f];
            var after = Fingerprint(f.GetValue(model));

            // Unreadable shapes and fields that are null both times tell us nothing.
            if (before is null && after is null) continue;
            if (before == after) stale.Add(f.Name);
        }

        Assert.True(stale.Count == 0,
            $"{model.GetType().Name}: {string.Join(", ", stale)} " +
            (stale.Count == 1 ? "is" : "are") + " marked [Scratch] but did not change across two " +
            "different forward passes. Scratch state is recomputed every pass by definition, so a " +
            "field that survives one unchanged is not scratch — most likely it is a WEIGHT, and " +
            "marking it [Scratch] removed it from ParameterCount and from GetParameters together. " +
            "That pair of omissions is invisible to the parameter-count contract test, because the " +
            "count and the vector agree on the same wrong answer. Reclassify it: declare it through " +
            "GetExtraTrainableTensors / RegisterComponents if it is trained, or mark it [Buffer] if " +
            "it is persistent state that is never optimized.");
    }
}
