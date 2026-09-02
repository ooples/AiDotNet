using System.Globalization;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Validation;

namespace AiDotNet.Configuration;

/// <summary>Configures the branch ratios and inspiration mix of <see cref="RatioEvolutionSelectionPolicy{TGenome}"/>.</summary>
/// <remarks>
/// <para>
/// Parent selection is a three-way mixture whose branch probabilities are <see cref="ExplorationRatio"/>,
/// <see cref="ExploitationRatio"/>, and <see cref="EliteRatio"/>; they must be non-negative and sum to one within a
/// tolerance of 1e-9, which is validated when the options are snapshotted and again when a policy is constructed
/// from them. The exploration branch draws a uniformly random occupied cell, the exploitation branch draws
/// uniformly from the strongest <see cref="ExploitationEliteCount"/> candidates selected by
/// <see cref="ExploitationSource"/>, and the elite branch takes the target island's current best entry.
/// Inspirations are then assembled deterministically: the island best when <see cref="IncludeIslandBest"/> is set,
/// then <see cref="TopInspirationCount"/> highest-quality elites, then <see cref="DiverseInspirationCount"/> entries
/// chosen by greedy maximum-minimum cell distance from the parent, and finally a uniform random fill if the engine
/// asked for more inspirations than those rules produced.
/// </para>
/// <para><b>For Beginners:</b> Evolution has to keep balancing two urges: try something new, or build on what
/// already works. These settings are that balance dial. The exploration share picks a parent at random so odd
/// corners of the search keep getting attention, the exploitation share picks one of the strongest candidates found
/// so far, and the elite share always starts from the current champion of the island being filled. The three shares
/// are probabilities, so they must add up to 1 (the defaults 0.2, 0.7, and 0.1 mirror OpenEvolve). The inspiration
/// settings then decide which other solutions are shown to your variation operator as extra ideas: a few of the very
/// best, plus a few deliberately different ones so the operator does not only ever see near-copies.</para>
/// </remarks>
public sealed class EvolutionSelectionOptions
{
    /// <summary>The largest absolute deviation from one that the three branch ratios may sum to.</summary>
    internal const double RatioSumTolerance = 1e-9;

    /// <summary>Gets or sets the probability of drawing a uniformly random occupied cell as the parent.</summary>
    public double ExplorationRatio { get; set; } = 0.2;

    /// <summary>Gets or sets the probability of drawing the parent from the configured elite pool.</summary>
    public double ExploitationRatio { get; set; } = 0.7;

    /// <summary>Gets or sets the probability of using the target island's best entry as the parent.</summary>
    public double EliteRatio { get; set; } = 0.1;

    /// <summary>Gets or sets which elite pool the exploitation branch draws from.</summary>
    public EvolutionExploitationSource ExploitationSource { get; set; } = EvolutionExploitationSource.GlobalTopK;

    /// <summary>Gets or sets how many strongest candidates the exploitation branch draws uniformly from.</summary>
    public int ExploitationEliteCount { get; set; } = 10;

    /// <summary>Gets or sets how many highest-quality elites are offered as inspirations.</summary>
    public int TopInspirationCount { get; set; } = 1;

    /// <summary>Gets or sets how many deliberately distant elites are offered as inspirations.</summary>
    public int DiverseInspirationCount { get; set; } = 2;

    /// <summary>Gets or sets whether the target island's best entry always leads the inspiration list.</summary>
    public bool IncludeIslandBest { get; set; } = true;

    /// <summary>Validates every value and returns an independent copy.</summary>
    /// <returns>A defensive copy that later mutation of this instance cannot affect.</returns>
    /// <exception cref="ArgumentOutOfRangeException">A ratio or count is outside its permitted range.</exception>
    /// <exception cref="ArgumentException">The three branch ratios do not sum to one.</exception>
    internal EvolutionSelectionOptions SnapshotAndValidate()
    {
        ValidateRatio(ExplorationRatio, nameof(ExplorationRatio));
        ValidateRatio(ExploitationRatio, nameof(ExploitationRatio));
        ValidateRatio(EliteRatio, nameof(EliteRatio));
        if (Math.Abs(ExplorationRatio + ExploitationRatio + EliteRatio - 1.0) > RatioSumTolerance)
            throw new ArgumentException(
                "ExplorationRatio, ExploitationRatio, and EliteRatio must sum to one.", nameof(ExplorationRatio));
        if (!Enum.IsDefined(typeof(EvolutionExploitationSource), ExploitationSource))
            throw new ArgumentOutOfRangeException(nameof(ExploitationSource));
        Guard.Positive(ExploitationEliteCount);
        if (TopInspirationCount < 0) throw new ArgumentOutOfRangeException(nameof(TopInspirationCount));
        if (DiverseInspirationCount < 0) throw new ArgumentOutOfRangeException(nameof(DiverseInspirationCount));

        return new EvolutionSelectionOptions
        {
            ExplorationRatio = ExplorationRatio,
            ExploitationRatio = ExploitationRatio,
            EliteRatio = EliteRatio,
            ExploitationSource = ExploitationSource,
            ExploitationEliteCount = ExploitationEliteCount,
            TopInspirationCount = TopInspirationCount,
            DiverseInspirationCount = DiverseInspirationCount,
            IncludeIslandBest = IncludeIslandBest
        };
    }

    /// <summary>Returns a stable, culture-independent representation suitable for compatibility hashes.</summary>
    /// <returns>The canonical text form of every value that changes selection behaviour.</returns>
    internal string ToCanonicalString() => string.Join("|", new[]
    {
        ExplorationRatio.ToString("R", CultureInfo.InvariantCulture),
        ExploitationRatio.ToString("R", CultureInfo.InvariantCulture),
        EliteRatio.ToString("R", CultureInfo.InvariantCulture),
        ((int)ExploitationSource).ToString(CultureInfo.InvariantCulture),
        ExploitationEliteCount.ToString(CultureInfo.InvariantCulture),
        TopInspirationCount.ToString(CultureInfo.InvariantCulture),
        DiverseInspirationCount.ToString(CultureInfo.InvariantCulture),
        IncludeIslandBest ? "island-best" : "no-island-best"
    });

    private static void ValidateRatio(double value, string parameterName)
    {
        if (!EvolutionDescriptorDefinition.IsFinite(value) || value < 0 || value > 1)
            throw new ArgumentOutOfRangeException(parameterName, "Selection ratios must be finite values in [0, 1].");
    }
}
