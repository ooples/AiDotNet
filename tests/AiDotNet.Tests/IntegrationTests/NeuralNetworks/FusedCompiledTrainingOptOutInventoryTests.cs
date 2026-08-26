using System.Reflection;
using AiDotNet.NeuralNetworks;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Pins which models opt out of fused compiled training, so a new opt-out has to be a decision
/// somebody made on purpose rather than a workaround that quietly accumulates.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why an inventory and not a rule.</b> <c>SupportsFusedCompiledTraining => false</c> is the
/// cheapest possible response to a model that misbehaves under compile-once replay, and it is almost
/// always the wrong one: this branch alone collected three such overrides — TableTransformer, PSENet
/// and BiaffineNER — whose justifications turned out to be one defect stated three ways ("live
/// aliases are not stable under compile-once replay"). Deleting them and fixing the cause is what
/// made the RG-LRU family fuse finitely with no opt-out at all.
/// </para>
/// <para>
/// A rule ("stateful layers must opt out") cannot express this: 53 of the 54 layers marked
/// <c>IsStateful</c> are fused-eligible today and are fine, because BatchNorm-style running state is
/// not the same problem as data-dependent control flow. So the honest guard is a census. The list
/// below is the set that existed when the flag was pruned back to master's condition; every entry is
/// a known-open item rather than an endorsement.
/// </para>
/// <para>
/// <b>If this test fails, do not just edit the list.</b> Growing it means a model was made to skip a
/// path instead of being fixed — the reviewer should ask for the underlying defect. Shrinking it is
/// the goal, and updating the list is then the correct and expected change.
/// </para>
/// </remarks>
public sealed class FusedCompiledTrainingOptOutInventoryTests
{
    /// <summary>
    /// The member the census counts. If it is ever renamed, this test starts reporting "no type
    /// declares this" and would pass vacuously, so the rename must come here too -- the removal
    /// assertion below is what makes that visible rather than silent.
    /// </summary>
    private const string PropertyName = "SupportsFusedCompiledTraining";

    /// <summary>
    /// The models that declared their own <c>SupportsFusedCompiledTraining => false</c> override at
    /// the time the per-layer flag was removed. Each is a candidate for the same treatment the
    /// RG-LRU family got: fix the replay defect, then delete the override.
    /// </summary>
    private static readonly string[] KnownOptOuts =
    {
        "Autoformer",
        "DifferentiableNeuralComputer",
        "DistilBERTNER",
        "GLALanguageModel",
        "GatedDeltaNetLanguageModel",
        "Hippo",
        "NeuralTuringMachine",
        "RWKVForecaster",
        "SAM2",
        "VideoLLaMA2",
    };

    [Fact]
    public void FusedCompiledTrainingOptOuts_MatchTheReviewedInventory()
    {
        var assembly = typeof(NeuralNetworkBase<double>).Assembly;

        var declaringTypes = assembly
            .GetTypes()
            .Where(type => type.GetProperty(
                    PropertyName,
                    BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public
                        | BindingFlags.DeclaredOnly) is not null)
            .Select(type => type.Name)
            // Generic model types arrive as "SAM2`1"; compare on the readable name.
            .Select(name => name.Contains('`') ? name[..name.IndexOf('`')] : name)
            .Distinct(StringComparer.Ordinal)
            // The base class declares the virtual itself; only overrides are interesting.
            .Where(name => !string.Equals(name, "NeuralNetworkBase", StringComparison.Ordinal))
            .OrderBy(name => name, StringComparer.Ordinal)
            .ToArray();

        var expected = KnownOptOuts.OrderBy(name => name, StringComparer.Ordinal).ToArray();

        var added = declaringTypes.Except(expected, StringComparer.Ordinal).ToArray();
        var removed = expected.Except(declaringTypes, StringComparer.Ordinal).ToArray();

        Assert.True(
            added.Length == 0,
            "These models newly opt out of fused compiled training: " + string.Join(", ", added)
            + ". An opt-out skips the optimized path for every caller of that model rather than "
            + "fixing what breaks under compile-once replay, and this branch already deleted three "
            + "overrides that were one defect described three ways. Fix the replay defect, or add "
            + "the model here with the reason if it is genuinely unfixable.");

        Assert.True(
            removed.Length == 0,
            "These models no longer opt out: " + string.Join(", ", removed)
            + ". That is the goal — remove them from KnownOptOuts to record the progress.");
    }
}
