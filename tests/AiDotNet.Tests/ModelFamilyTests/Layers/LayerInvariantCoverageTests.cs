using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Interfaces;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

/// <summary>
/// Bounds the number of layer types that no invariant test covers, so the count cannot quietly grow.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="LayerTestBase{T}"/> asserts eleven properties every layer must satisfy, and for a long time
/// nothing derived from it: the invariants ran against zero real layers while the README advertised the
/// subsystem as stable on "bottom-up invariant tests at 94% pass rate". Wiring the core layers in fixed the
/// present, but nothing stopped the situation recurring the next time somebody adds a layer.
/// </para>
/// <para>
/// A floor on the number of COVERED layers would not stop it. Adding an uncovered layer leaves the covered
/// count untouched, so that test stays green while coverage as a proportion falls — the same hole the
/// documentation gate had, where a pass-count floor could not see a newly added broken example. Bounding
/// the UNCOVERED count is what actually bites: add a layer without an invariant subclass and this fails.
/// </para>
/// <para>
/// Lower <see cref="MaxUncoveredLayers"/> whenever coverage improves. Raising it is the thing to argue
/// about in review.
/// </para>
/// </remarks>
public sealed class LayerInvariantCoverageTests
{
    private readonly ITestOutputHelper _output;

    public LayerInvariantCoverageTests(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// Layer types with no <see cref="LayerTestBase{T}"/> subclass. Ratchet this DOWN as layers are wired in.
    /// </summary>
    private const int MaxUncoveredLayers = 121;

    [Fact]
    [Trait("category", "integration-configure-method")]
    public void Adding_a_layer_without_an_invariant_subclass_is_not_allowed()
    {
        var layerTypes = ConcreteLayerTypes().ToList();
        var covered = CoveredLayerTypes();
        var uncovered = layerTypes.Where(t => !covered.Contains(t)).ToList();

        _output.WriteLine($"layer types: {layerTypes.Count}, covered: {covered.Count}, uncovered: {uncovered.Count}");
        foreach (var t in uncovered.Take(20).OrderBy(t => t.Name, StringComparer.Ordinal))
        {
            _output.WriteLine($"  uncovered: {t.Name}");
        }

        Assert.True(
            uncovered.Count <= MaxUncoveredLayers,
            $"{uncovered.Count} layer types have no LayerTestBase subclass, above the recorded ceiling of " +
            $"{MaxUncoveredLayers}. A layer was added without wiring it into the invariant harness. Adding one " +
            "costs about four lines in CoreLayerInvariantTests.cs — give it an InputShape and a CreateLayer() " +
            "— then lower the ceiling. The harness checks finite output, deterministic replay, shape " +
            "agreement, parameter roundtrip, serialization and gradient correctness, so wiring a layer in is " +
            "the cheapest coverage available.\n  " +
            string.Join("\n  ", uncovered.Select(t => t.Name).OrderBy(n => n, StringComparer.Ordinal).Take(30)));
    }

    /// <summary>Every concrete, publicly constructible layer in the library, as open generic definitions.</summary>
    private static IEnumerable<Type> ConcreteLayerTypes()
    {
        var assembly = typeof(ILayer<>).Assembly;
        foreach (var type in assembly.GetTypes())
        {
            if (!type.IsClass || type.IsAbstract || !type.IsPublic) continue;
            if (!type.GetInterfaces().Any(i => i.IsGenericType &&
                                               i.GetGenericTypeDefinition() == typeof(ILayer<>)))
            {
                continue;
            }
            yield return type.IsGenericType ? type.GetGenericTypeDefinition() : type;
        }
    }

    /// <summary>
    /// The layer each invariant subclass builds, found by invoking its CreateLayer(). Asking the subclass
    /// what it constructs is exact, where matching on class names would drift the moment one is renamed.
    /// </summary>
    private static HashSet<Type> CoveredLayerTypes()
    {
        var covered = new HashSet<Type>();
        foreach (var testType in typeof(LayerInvariantCoverageTests).Assembly.GetTypes())
        {
            if (testType.IsAbstract || !DerivesFromLayerTestBase(testType)) continue;

            var create = testType.GetMethod("CreateLayer", BindingFlags.Instance | BindingFlags.NonPublic);
            if (create is null) continue;

            try
            {
                if (Activator.CreateInstance(testType) is not { } instance) continue;
                if (create.Invoke(instance, null) is not { } layer) continue;
                var t = layer.GetType();
                covered.Add(t.IsGenericType ? t.GetGenericTypeDefinition() : t);
            }
            catch
            {
                // A fixture that cannot be built here still has its own tests; it just cannot be credited.
            }
        }
        return covered;
    }

    private static bool DerivesFromLayerTestBase(Type type)
    {
        for (var t = type.BaseType; t is not null; t = t.BaseType)
        {
            if (t.IsGenericType && t.GetGenericTypeDefinition() == typeof(LayerTestBase<>)) return true;
        }
        return false;
    }
}
