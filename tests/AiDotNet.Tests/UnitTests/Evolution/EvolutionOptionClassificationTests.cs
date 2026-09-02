using System.Reflection;
using AiDotNet.Configuration;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution;

public sealed class EvolutionOptionClassificationTests
{
    // The deliberate classification of every EvolutionEngineOptions property. Semantic options change what the search
    // means and are hashed into the resume compatibility identity; budget options only bound or locate a run and are
    // recorded for provenance without ever being compared. Anything not listed here is treated as semantic, so a new
    // option that nobody classified fails these tests rather than silently escaping the compatibility hash.
    private static readonly HashSet<string> BudgetOptionNames = new(StringComparer.Ordinal)
    {
        nameof(EvolutionEngineOptions.RunId),
        nameof(EvolutionEngineOptions.OutputDirectory),
        nameof(EvolutionEngineOptions.MaxEvaluationAttempts),
        nameof(EvolutionEngineOptions.MaxProposals),
        nameof(EvolutionEngineOptions.MaxGenerations),
        nameof(EvolutionEngineOptions.TimeLimit),
        nameof(EvolutionEngineOptions.CheckpointInterval),
        nameof(EvolutionEngineOptions.Resume),
        nameof(EvolutionEngineOptions.MaxDegreeOfParallelism)
    };

    // The third category, and the only one allowed to appear in NEITHER canonical string. A derived option does not
    // reach the compatibility hash itself because something it selects already does, so hashing it as well would
    // wrongly refuse a resume between two runs that behave identically.
    //
    // SelectionPolicy only takes effect when the caller supplies no ISelectionPolicy, and the engine then folds the
    // policy it built - Id plus VersionHash, and a ratio policy's VersionHash already carries Selection's own
    // canonical string - into the compatibility hash. Two engines that differ only in this field but end up running
    // the same policy are genuinely interchangeable for resume, and EvolutionEngineReachabilityTests
    // .AnExplicitlySuppliedSelectionPolicyWinsOverTheOptions pins exactly that: an explicit Uniform policy matches a
    // pure-uniform engine's CompatibilityHash while both differ from the options-built Ratio engine's.
    //
    // Membership here must be argued, never convenient: an option belongs only if changing it provably moves the
    // engine's CompatibilityHash by another route. An unclassified option still fails, which is the point.
    private static readonly HashSet<string> DerivedOptionNames = new(StringComparer.Ordinal)
    {
        nameof(EvolutionEngineOptions.SelectionPolicy)
    };

    [Fact]
    public void EveryEngineOptionLandsOnExactlyOneSideOfTheSemanticBudgetSplit()
    {
        PropertyInfo[] properties = WritableOptions();
        Assert.NotEmpty(properties);

        foreach (PropertyInfo property in properties)
        {
            var options = new EvolutionEngineOptions();
            string semanticBefore = options.ToSemanticCanonicalString();
            string budgetBefore = options.ToBudgetCanonicalString();

            property.SetValue(options, AlternativeValue(property, property.GetValue(options)));

            bool semanticChanged =
                !string.Equals(semanticBefore, options.ToSemanticCanonicalString(), StringComparison.Ordinal);
            bool budgetChanged =
                !string.Equals(budgetBefore, options.ToBudgetCanonicalString(), StringComparison.Ordinal);

            if (DerivedOptionNames.Contains(property.Name))
            {
                Assert.False(semanticChanged || budgetChanged,
                    $"'{property.Name}' is listed as derived, which means it must reach the compatibility hash only " +
                    "through what it selects. It now writes a canonical field of its own, so either drop it from " +
                    "DerivedOptionNames or stop recording it.");
                continue;
            }

            Assert.True(semanticChanged || budgetChanged,
                $"'{property.Name}' is recorded by neither SemanticFields() nor BudgetFields(), so changing it would " +
                "silently leave the compatibility hash untouched and let an incompatible checkpoint resume. Add it to " +
                "one of the two lists, or to DerivedOptionNames with an argument for why another value already " +
                "carries its effect into the hash.");
            Assert.False(semanticChanged && budgetChanged,
                $"'{property.Name}' is recorded by both SemanticFields() and BudgetFields(); an option belongs to " +
                "exactly one side of the split.");
            Assert.Equal(BudgetOptionNames.Contains(property.Name), budgetChanged);
        }
    }

    [Fact]
    public void ChangingAnySemanticOptionChangesTheConfigurationHashThatGuardsResume()
    {
        string baseline = new EvolutionEngineOptions().ToSemanticCanonicalString();
        var seen = new HashSet<string>(StringComparer.Ordinal);

        foreach (PropertyInfo property in WritableOptions())
        {
            if (BudgetOptionNames.Contains(property.Name)) continue;
            if (DerivedOptionNames.Contains(property.Name)) continue;
            var options = new EvolutionEngineOptions();
            property.SetValue(options, AlternativeValue(property, property.GetValue(options)));
            string canonical = options.ToSemanticCanonicalString();

            Assert.NotEqual(baseline, canonical);
            Assert.True(seen.Add(canonical),
                $"'{property.Name}' produces a canonical string another option already produces, so the two are " +
                "indistinguishable to a resume check.");
        }
    }

    [Fact]
    public void SnapshotAndValidateCarriesEveryEngineOptionIntoItsDefensiveCopy()
    {
        foreach (PropertyInfo property in WritableOptions())
        {
            var options = new EvolutionEngineOptions();
            // A grace period is only valid alongside a timeout, so this pair is always configured together.
            if (property.Name == nameof(EvolutionEngineOptions.EvaluationGracePeriod))
                options.EvaluationTimeout = TimeSpan.FromSeconds(11);
            property.SetValue(options, AlternativeValue(property, property.GetValue(options)));

            EvolutionEngineOptions snapshot = options.SnapshotAndValidate();

            Assert.Equal(options.ToSemanticCanonicalString(), snapshot.ToSemanticCanonicalString());
            Assert.Equal(options.ToBudgetCanonicalString(), snapshot.ToBudgetCanonicalString());

            // A derived option writes no canonical field, so the two comparisons above would pass even if the
            // defensive copy dropped it entirely and the engine silently fell back to the default policy. Compare the
            // value itself for those.
            if (DerivedOptionNames.Contains(property.Name))
                Assert.Equal(property.GetValue(options), property.GetValue(snapshot));
        }
    }

    private static PropertyInfo[] WritableOptions() => typeof(EvolutionEngineOptions)
        .GetProperties(BindingFlags.Public | BindingFlags.Instance)
        .Where(property => property.CanRead && property.CanWrite && property.GetIndexParameters().Length == 0)
        .OrderBy(property => property.Name, StringComparer.Ordinal)
        .ToArray();

    private static object AlternativeValue(PropertyInfo property, object? current)
    {
        // An absolute path so that the snapshot's own Path.GetFullPath normalization is a no-op.
        if (property.Name == nameof(EvolutionEngineOptions.OutputDirectory))
            return Path.GetFullPath("evolution-option-classification");

        Type type = Nullable.GetUnderlyingType(property.PropertyType) ?? property.PropertyType;
        if (type == typeof(string)) return "alternative";
        if (type == typeof(bool)) return !(current is bool flag && flag);
        if (type == typeof(int)) return (current is int number ? number : 0) + 7;
        if (type == typeof(long)) return (current is long count ? count : 0L) + 7L;
        if (type == typeof(ulong)) return (current is ulong seed ? seed : 0UL) + 7UL;
        // Every double option is valid at least half a unit above its default: the rate stays inside [0,1], the retry
        // multiplier stays at or above one, and the novelty threshold and target quality stay finite.
        if (type == typeof(double)) return (current is double value ? value : 0d) + 0.5d;
        if (type == typeof(TimeSpan))
            return (current is TimeSpan duration ? duration : TimeSpan.Zero) + TimeSpan.FromSeconds(7);
        if (type.IsEnum)
        {
            foreach (object candidate in Enum.GetValues(type))
                if (!Equals(candidate, current)) return candidate;
            throw new InvalidOperationException($"'{property.Name}' declares only one value, so it cannot be varied.");
        }
        if (type == typeof(EvolutionSelectionOptions))
            return new EvolutionSelectionOptions { TopInspirationCount = 5 };
        if (type == typeof(EvolutionCascadeOptions))
            return new EvolutionCascadeOptions { ChargeRejectedStagesToBudget = true };
        if (type == typeof(EvolutionArtifactOptions))
            return new EvolutionArtifactOptions { MaxArtifactBytes = 4_096 };
        if (type == typeof(EvolutionEarlyStoppingOptions))
            return new EvolutionEarlyStoppingOptions { PatienceEvaluations = 9 };

        throw new InvalidOperationException(
            $"'{property.Name}' has type '{type}', which this test cannot vary. Extend AlternativeValue so the new " +
            "option is still checked against the semantic/budget split.");
    }
}
