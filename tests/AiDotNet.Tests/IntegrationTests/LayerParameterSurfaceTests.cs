using System;
using System.Collections.Generic;
using System.Globalization;
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
        // EVERY VISITED LAYER LANDS IN A BUCKET. A layer with no public GetParameters() used to
        // be skipped with a bare continue, incrementing nothing, so the summary reported a total
        // that silently omitted it -- inflating coverage in exactly the way the counting exists
        // to prevent.
        int checkedCount = 0, unconstructable = 0, unsized = 0, noParameterApi = 0;

        var logPath = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "layer-parameter-surface.txt");
        // DISPOSED ON EVERY PATH, AND A FAILURE TO OPEN IS REPORTED. The manual dispose ran only
        // on the normal path, so a throw from the enumeration below -- Assembly.GetTypes() raises
        // ReflectionTypeLoadException, and this sweep calls it -- leaked the handle and left the
        // partial log the comment above calls "the deliverable" unflushed. The empty catch was the
        // other half: when the file could not be opened at all, every later write silently went
        // nowhere and the run looked like one that simply found nothing.
        System.IO.StreamWriter? log = null;
        try
        {
            log = new System.IO.StreamWriter(logPath, append: false) { AutoFlush = true };
        }
        catch (Exception ex)
        {
            _output.WriteLine($"NOTE: could not open {logPath} ({ex.GetType().Name}: {ex.Message}); " +
                              "the console output below is the only record of this run.");
        }

        using var _logHandle = log;

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
                if (m is null) { noParameterApi++; continue; }
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

        _output.WriteLine($"Checked {checkedCount} layers; {unsized} agree at zero (deferred); " +
                          $"{noParameterApi} expose no public GetParameters(); " +
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

    /// <summary>
    /// Builds a layer for measurement, preferring a default constructor and falling back to the
    /// constructor arguments the layer already declares for scaffold generation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Requiring a parameterless constructor reached 39 layers. Every composite this invariant is
    /// actually about — ClozeAttention, BranchformerBlock, ConformerBlock, VideoGigaGAN, the ResNet
    /// blocks — takes its widths as constructor arguments, so all of them sat outside the sweep and
    /// it reported agreement across a set that excluded them. That is the worse failure: a green
    /// sweep over the layers that were never at risk reads exactly like a green sweep over all of
    /// them.
    /// </para>
    /// <para>
    /// The widths are not guessed here. <c>[LayerProperty(TestConstructorArgs = "...")]</c> already
    /// states them on the layer, and the test scaffold generator emits real constructor calls from
    /// that same string; this replays it through reflection.
    /// </para>
    /// <para>
    /// Only all-numeric argument lists can be replayed — the declaration is C# source, so an entry
    /// like <c>new int[] { 8, 12 }</c> has no reflection equivalent. Those layers stay unmeasured
    /// and are COUNTED as unconstructable, because a sweep that silently dropped them would report
    /// the same clean result whether a layer agreed or was simply never asked.
    /// </para>
    /// </remarks>
    private static object? TryConstruct(Type closed)
    {
        var ctor = closed.GetConstructors(BindingFlags.Public | BindingFlags.Instance)
            .Where(c => c.GetParameters().All(p => p.HasDefaultValue) || c.GetParameters().Length == 0)
            .OrderBy(c => c.GetParameters().Length)
            .FirstOrDefault();
        if (ctor is not null)
        {
            var defaults = ctor.GetParameters()
                .Select(p => p.DefaultValue == DBNull.Value ? null : p.DefaultValue).ToArray();
            return ctor.Invoke(defaults);
        }

        var declared = DeclaredTestArguments(closed);
        if (declared is null) return null;

        foreach (var candidate in closed.GetConstructors(BindingFlags.Public | BindingFlags.Instance)
                     .OrderBy(c => c.GetParameters().Length))
        {
            var parameters = candidate.GetParameters();
            if (parameters.Length < declared.Length) continue;
            // Anything the declaration does not supply must carry its own default, or this overload
            // is not the one the declaration was written against.
            if (!parameters.Skip(declared.Length).All(p => p.HasDefaultValue)) continue;

            var args = new object?[parameters.Length];
            bool usable = true;
            for (int i = 0; i < parameters.Length; i++)
            {
                if (i >= declared.Length)
                {
                    args[i] = parameters[i].DefaultValue == DBNull.Value
                        ? null
                        : parameters[i].DefaultValue;
                    continue;
                }

                var target = Nullable.GetUnderlyingType(parameters[i].ParameterType)
                             ?? parameters[i].ParameterType;
                // Numeric primitives only. bool and char are primitives too, and silently turning
                // a declared 1 into true would construct a DIFFERENT layer than the declaration
                // describes and then measure it as though it were the right one.
                if (!target.IsPrimitive || target == typeof(bool) || target == typeof(char))
                {
                    usable = false;
                    break;
                }

                try { args[i] = Convert.ChangeType(declared[i], target, CultureInfo.InvariantCulture); }
                catch (Exception ex) when (ex is InvalidCastException or OverflowException or FormatException)
                {
                    usable = false;
                    break;
                }
            }

            if (usable) return candidate.Invoke(args);
        }

        return null;
    }

    /// <summary>True when the layer states constructor arguments for scaffold generation.</summary>
    private static bool DeclaresTestArguments(Type closed)
        => !string.IsNullOrWhiteSpace(TestConstructorArgs(closed));

    private static string? TestConstructorArgs(Type closed)
        => closed.GetCustomAttributes(inherit: false)
            .OfType<AiDotNet.Attributes.LayerPropertyAttribute>()
            .FirstOrDefault()?.TestConstructorArgs;

    /// <summary>
    /// The numeric constructor arguments a layer declares for scaffold generation, or <c>null</c>
    /// when it declares none or declares one reflection cannot reconstruct.
    /// </summary>
    private static double[]? DeclaredTestArguments(Type closed)
    {
        var args = TestConstructorArgs(closed);
        if (string.IsNullOrWhiteSpace(args)) return null;

        var tokens = args.Split(',');
        var values = new double[tokens.Length];
        for (int i = 0; i < tokens.Length; i++)
        {
            if (!double.TryParse(tokens[i].Trim(), NumberStyles.Float, CultureInfo.InvariantCulture,
                    out values[i]))
                return null;
        }
        return values;
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

            // Gated on DECLARING test arguments, not on their being replayable. A layer whose
            // declaration this sweep cannot reconstruct is a coverage MISS, and it has to land in
            // the "not constructable" bucket to be visible as one; filtering it out here instead
            // would drop it from the denominator, so the summary would report full coverage of a
            // set that had quietly shrunk to the layers that happened to be easy.
            if (!closed.GetConstructors(BindingFlags.Public | BindingFlags.Instance)
                    .Any(c => c.GetParameters().Length == 0 || c.GetParameters().All(p => p.HasDefaultValue))
                && !DeclaresTestArguments(closed))
                continue;

            yield return closed;
        }
    }
}
