using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using AiDotNet.Attributes;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.Cloning;

/// <summary>
/// Sweeps every layer type through the clone adapter to find which cannot be rebuilt.
/// </summary>
/// <remarks>
/// <para>
/// The coverage number explains how much the harness reached, while every reached layer is a hard
/// correctness gate. A layer whose required constructor arguments are not marked <c>[LayerState]</c>
/// has no generated factory, so it cannot be rebuilt — and the only way to know how many of those
/// there are is to try all of them.
/// </para>
/// <para>
/// Construction arguments come from <c>[LayerProperty(TestConstructorArgs = ...)]</c>, which 156 of
/// 189 layers already declare for test generation. That value is C# source text rather than runtime
/// metadata, so only simple numeric arguments can be coerced here; anything else is counted as
/// unconstructible-by-this-test and reported separately from a genuine clone failure. Conflating
/// the two would let a shortfall in the harness read as a shortfall in the feature.
/// </para>
/// </remarks>
public class AllLayersCloneTests
{
    private readonly ITestOutputHelper _output;

    /// <summary>Initializes a new instance of the <see cref="AllLayersCloneTests"/> class.</summary>
    /// <param name="output">Sink for the coverage summary.</param>
    public AllLayersCloneTests(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// Reports how many layer types can be rebuilt by the clone adapter.
    /// </summary>
    /// <returns>A task representing the test.</returns>
    [Fact(Timeout = 600000)]
    public async Task EveryLayer_ReportsWhetherItCanBeCloned()
    {
        await Task.Yield();

        var layerBase = typeof(LayerBase<>);
        var candidates = layerBase.Assembly.GetTypes()
            .Where(t => t.IsClass && !t.IsAbstract && !t.IsNested)
            .Where(t => t.IsGenericTypeDefinition && t.GetGenericArguments().Length == 1)
            .Where(t => DerivesFromLayerBase(t))
            .OrderBy(t => t.Name, StringComparer.Ordinal)
            .ToList();

        var cloned = new List<string>();
        var failed = new List<string>();
        var notConstructed = new List<string>();

        // COUNTED AND REPORTED, because the first attempt at forwarding silently did nothing and
        // produced a number identical to the unforwarded run. If this reads 0, the probe never
        // fired and the coverage figure below is measuring unresolved layers again.
        var forwarded = new List<string>();

        foreach (var open in candidates)
        {
            Type closed;
            try
            {
                closed = open.MakeGenericType(typeof(double));
            }
            catch (Exception)
            {
                notConstructed.Add($"{open.Name}: constraints reject double");
                continue;
            }

            var instance = TryConstruct(closed);
            if (instance is null)
            {
                notConstructed.Add($"{open.Name}: no usable TestConstructorArgs");
                continue;
            }

            try
            {
                // FORWARD FIRST. Cloning an unforwarded layer compares two unresolved layers that
                // trivially agree at zero parameters, which is why this sweep read 119/0 while the
                // trained-layer proof was failing. A layer that has been USED is the case worth
                // measuring.
                var typed = (LayerBase<double>)instance;
                if (Forward(typed)) forwarded.Add(open.Name);

                // LayerBase declares Clone as a public virtual instance method, which is the
                // surviving mechanism after #1789 replaced this branch's LayerCloning extension with
                // LayerStateGenerator's generated factory. The sweep itself is unchanged: construct
                // every layer, forward it, clone it, and require the clone to be the same type.
                var clone = typed.Clone();
                if (clone is null)
                {
                    failed.Add($"{open.Name}: clone returned null");
                    continue;
                }

                if (clone.GetType() != closed)
                {
                    failed.Add($"{open.Name}: clone is {clone.GetType().Name}");
                    continue;
                }

                cloned.Add(open.Name);
            }
            catch (Exception ex)
            {
                var message = (ex.InnerException ?? ex).Message;
                failed.Add($"{open.Name}: {(ex.InnerException ?? ex).GetType().Name}: "
                    + message.Substring(0, Math.Min(90, message.Length)));
            }
        }

        _output.WriteLine($"layer types        : {candidates.Count}");
        _output.WriteLine($"cloned OK          : {cloned.Count}");
        _output.WriteLine($"clone FAILED       : {failed.Count}");
        _output.WriteLine($"not constructed    : {notConstructed.Count} (harness limit, not a clone result)");
        _output.WriteLine($"forwarded first    : {forwarded.Count} of {cloned.Count + failed.Count} attempted");
        _output.WriteLine(string.Empty);

        foreach (var f in failed.Take(40)) _output.WriteLine("  FAIL  " + f);
        foreach (var n in notConstructed.Take(15)) _output.WriteLine("  skip  " + n);

        // A REPORT FILE, not just ITestOutputHelper. xunit surfaces the helper only on a failing
        // test or under `verbosity=detailed`, and detailed logs all 72,235 discovered cases -- 18MB
        // per run to read five lines out of. Nine parallel runs of that filled the system drive to
        // zero bytes free. Writing the summary here means the run needs no console logger at all.
        // AIDOTNET_SWEEP_DIR redirects it off the system drive when that drive is short.
        var dir = Environment.GetEnvironmentVariable("AIDOTNET_SWEEP_DIR");
        if (string.IsNullOrEmpty(dir)) dir = System.IO.Path.GetTempPath();

        var report = new List<string>
        {
            $"layer types        : {candidates.Count}",
            $"cloned OK          : {cloned.Count}",
            $"clone FAILED       : {failed.Count}",
            $"not constructed    : {notConstructed.Count} (harness limit, not a clone result)",
            $"forwarded first    : {forwarded.Count} of {cloned.Count + failed.Count} attempted",
            string.Empty,
        };
        report.AddRange(failed.Select(f => $"FAIL  {f}"));
        report.AddRange(notConstructed.Select(n => $"skip  {n}"));
        System.IO.File.WriteAllLines(
            System.IO.Path.Combine(dir, "aidotnet-layer-clone-sweep.txt"), report);

        // Coverage can grow without pinning a brittle count, but every layer the harness actually
        // reaches must clone. The previous measurement-only assertion let a non-zero failure list
        // produce a green test, which made the sweep documentation rather than regression proof.
        Assert.NotEmpty(cloned);
        Assert.True(
            failed.Count == 0,
            $"{failed.Count} constructed layer(s) failed cloning:{Environment.NewLine}"
                + string.Join(Environment.NewLine, failed));
    }

    private static bool DerivesFromLayerBase(Type type)
    {
        for (var b = type.BaseType; b is not null; b = b.BaseType)
        {
            if (b.IsGenericType && b.GetGenericTypeDefinition().Name == "LayerBase`1") return true;
            if (b.Name == "LayerBase`1") return true;
        }

        return false;
    }

    /// <summary>
    /// Builds a layer from its declared test constructor arguments, when they can be coerced.
    /// </summary>
    /// <returns>The instance, or null when this harness cannot supply the arguments.</returns>
    /// <remarks>
    /// Only simple numeric literals are handled. Returning null rather than guessing keeps an
    /// unconstructible layer out of the failure count, since being unable to build a layer here
    /// says nothing about whether it clones.
    /// </remarks>
    /// <summary>Pushes one probe through the layer so a lazy width resolves. True if it ran.</summary>
    /// <remarks>
    /// <para>
    /// The declared shape CANNOT be used as the probe. A lazy layer declares <c>[-1]</c> for the
    /// axis it has not resolved yet, so a guard of <c>shape[0] > 0</c> skips precisely the layers
    /// that needed forwarding, and the sweep reports on unresolved layers while appearing to have
    /// forwarded them. That mistake cost two runs -- the same wrong assumption that made
    /// <c>ResolveShapesOnly</c> a no-op.
    /// </para>
    /// <para>
    /// So every non-positive axis becomes a small concrete size, and both shape conventions are
    /// tried: layers whose declared shape excludes the batch axis, and layers whose shape includes
    /// it. The first probe that does not throw wins.
    /// </para>
    /// </remarks>
    private static bool Forward(LayerBase<double> layer)
    {
        int[] declared;
        try
        {
            declared = layer.GetInputShape();
        }
        catch (Exception)
        {
            return false;
        }

        if (declared is null || declared.Length == 0) return false;

        var concrete = new int[declared.Length];
        for (var i = 0; i < declared.Length; i++) concrete[i] = declared[i] > 0 ? declared[i] : 4;

        // Batch-prefixed first: GetInputShape describes ONE sample for most layers here.
        var batched = new int[concrete.Length + 1];
        batched[0] = 1;
        Array.Copy(concrete, 0, batched, 1, concrete.Length);

        foreach (var probe in new[] { batched, concrete })
        {
            try
            {
                layer.Forward(new Tensor<double>(probe));
                return true;
            }
            catch (Exception)
            {
                // Try the other convention; a layer that refuses both is measured unforwarded.
            }
        }

        return false;
    }

    private static object? TryConstruct(Type closed)
    {
        var attribute = closed.GetCustomAttributes(inherit: false)
            .OfType<LayerPropertyAttribute>()
            .FirstOrDefault();

        var raw = attribute?.TestConstructorArgs;
        if (string.IsNullOrWhiteSpace(raw)) return null;

        var literals = raw!.Split(',').Select(s => s.Trim()).ToArray();
        if (literals.Any(l => !int.TryParse(l, NumberStyles.Integer, CultureInfo.InvariantCulture, out _)))
        {
            return null;
        }

        var values = literals
            .Select(l => int.Parse(l, NumberStyles.Integer, CultureInfo.InvariantCulture))
            .ToArray();

        foreach (var ctor in closed.GetConstructors().OrderBy(c => c.GetParameters().Length))
        {
            var parameters = ctor.GetParameters();
            if (parameters.Length < values.Length) continue;
            if (parameters.Take(values.Length).Any(p => p.ParameterType != typeof(int))) continue;
            if (parameters.Skip(values.Length).Any(p => !p.IsOptional)) continue;

            var args = new object?[parameters.Length];
            for (int i = 0; i < values.Length; i++) args[i] = values[i];
            for (int i = values.Length; i < parameters.Length; i++) args[i] = Type.Missing;

            try
            {
                return ctor.Invoke(BindingFlags.OptionalParamBinding, binder: null, args, culture: null);
            }
            catch (Exception)
            {
                // Try the next overload rather than declaring the layer unconstructible.
            }
        }

        return null;
    }
}
