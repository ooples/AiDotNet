using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Attributes;
using AiDotNet.Audio.Whisper;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// Invariants every <c>[PaperOptimizer]</c> declaration in the library must satisfy (#1928).
/// </summary>
/// <remarks>
/// <para>
/// These catch the failure mode that matters most while ~850 declarations are being added: a
/// declaration that LOOKS authoritative and is never selected. The analyzer catches what it can at
/// compile time, but it cannot see whether a variant key can ever match, because that depends on a
/// runtime property. A dead declaration is worse than a missing one -- a missing recipe reports
/// NotDeclared, whereas a dead one reports nothing at all while the model quietly trains on library
/// defaults, which is precisely the defect this issue exists to end.
/// </para>
/// </remarks>
public class PaperRecipeDeclarationHygieneTests
{
    private static IEnumerable<(Type Type, PaperOptimizerAttribute[] Rows)> Declarations()
    {
        Type[] types;
        try { types = typeof(PaperOptimizerAttribute).Assembly.GetTypes(); }
        catch (ReflectionTypeLoadException ex) { types = ex.Types.OfType<Type>().ToArray(); }

        foreach (Type type in types)
        {
            var rows = (PaperOptimizerAttribute[])type.GetCustomAttributes(
                typeof(PaperOptimizerAttribute), inherit: false);
            if (rows.Length > 0) yield return (type, rows);
        }
    }

    [Fact]
    public void EveryVariantKeyedDeclarationIsOnATypeThatCanReportAVariant()
    {
        // A Variant row on a type that does not implement IPaperOptimizerVariant can never be
        // selected: resolution reads the variant from the instance, gets null, and falls back to the
        // unkeyed row. The declaration would sit in the source looking like a cited fact forever.
        var dead = Declarations()
            .Where(d => d.Rows.Any(r => r.Variant.Length > 0))
            .Where(d => !typeof(IPaperOptimizerVariant).IsAssignableFrom(d.Type))
            .Select(d => d.Type.Name)
            .ToList();

        Assert.True(dead.Count == 0,
            "these types declare variant-keyed recipes but cannot report a variant, so those rows "
            + "can never be selected: " + string.Join(", ", dead));
    }

    [Fact]
    public void EveryDeclarationThatStatesValuesCitesASource()
    {
        // Source is the anti-fabrication guard. AIDN102 enforces it when compiling from source;
        // this re-checks the built assembly, which is what actually ships.
        var uncited = Declarations()
            .SelectMany(d => d.Rows.Select(r => (d.Type, Row: r)))
            .Where(x => x.Row.DeclaresAnyHyperparameter && string.IsNullOrWhiteSpace(x.Row.Source))
            .Select(x => x.Type.Name)
            .ToList();

        Assert.True(uncited.Count == 0,
            "these declare hyperparameters with no Source: " + string.Join(", ", uncited));
    }

    [Fact]
    public void NoTypeDeclaresTheSameVariantAndComponentTwice()
    {
        // Two rows with the same key are indistinguishable to resolution, so one of them silently
        // loses. Which one is an accident of attribute ordering.
        var duplicated = new List<string>();
        foreach (var (type, rows) in Declarations())
        {
            var keys = rows.Select(r => r.Variant + "|" + r.Component).ToList();
            if (keys.Count != keys.Distinct(StringComparer.OrdinalIgnoreCase).Count())
                duplicated.Add(type.Name);
        }

        Assert.True(duplicated.Count == 0,
            "these declare duplicate variant/component keys: " + string.Join(", ", duplicated));
    }

    [Fact]
    public void WhisperDeclaresOneRowPerSizeThePaperGivesARateFor()
    {
        // Radford et al. 2022 Table 19 gives a max learning rate for six sizes and none for
        // LargeV3, which post-dates the paper. LargeV3 must therefore fall through to the unkeyed
        // row -- which carries Table 17's shared settings and deliberately no learning rate.
        var rows = (PaperOptimizerAttribute[])typeof(WhisperModel<double>)
            .GetCustomAttributes(typeof(PaperOptimizerAttribute), inherit: false);

        var keyed = rows.Where(r => r.Variant.Length > 0).ToList();
        Assert.Equal(6, keyed.Count);
        Assert.DoesNotContain(nameof(WhisperModelSize.LargeV3), keyed.Select(r => r.Variant));

        // Every keyed row names a real size, so a renamed enum member cannot leave a stale key.
        foreach (var row in keyed)
            Assert.True(Enum.IsDefined(typeof(WhisperModelSize), row.Variant), row.Variant);

        var fallback = Assert.Single(rows.Where(r => r.Variant.Length == 0));
        Assert.True(double.IsNaN(fallback.LearningRate),
            "the fallback row must not state a learning rate the paper does not give for LargeV3");

        // The shared Table 17 values must be on every row, since resolution returns one row and
        // does not merge -- omitting them from the keyed rows would silently drop the recipe.
        foreach (var row in rows)
        {
            Assert.Equal(OptimizerKind.AdamW, row.Optimizer);
            Assert.Equal(0.98, row.Beta2, precision: 12);
            Assert.Equal(0.1, row.WeightDecay, precision: 12);
            Assert.Equal(2048, row.WarmupSteps);
            Assert.Equal(1.0, row.MaxGradientNorm, precision: 12);
            Assert.Equal(256, row.ReferenceBatchSize);
        }
    }
}
