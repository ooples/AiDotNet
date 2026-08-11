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
/// The point is the coverage number, not a pass. A layer whose required constructor arguments are
/// not marked <c>[LayerState]</c> has no generated factory, so it cannot be rebuilt — and the only
/// way to know how many of those there are is to try all of them.
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
                try
                {
                    var shape = typed.GetInputShape();
                    if (shape.Length > 0 && shape[0] > 0) typed.Forward(new Tensor<double>(shape));
                }
                catch (Exception)
                {
                    // A layer that will not take a probe of its own declared shape is measured
                    // unforwarded, the same as before.
                }

                var clone = LayerCloning.Clone(typed);
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
        _output.WriteLine(string.Empty);

        foreach (var f in failed.Take(40)) _output.WriteLine("  FAIL  " + f);
        foreach (var n in notConstructed.Take(15)) _output.WriteLine("  skip  " + n);

        // The sweep is a measurement first. Asserting only that SOMETHING was exercised keeps a
        // harness that constructs nothing from reporting success, without pinning a number that
        // will move as layers gain [LayerState] coverage.
        Assert.True(
            cloned.Count + failed.Count > 0,
            "No layer was constructed, so this run measured nothing.");
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
