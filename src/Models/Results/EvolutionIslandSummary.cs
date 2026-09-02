namespace AiDotNet.Models.Results;

/// <summary>How one island of an evolution run ended: its coverage, its best entry, and how far it advanced.</summary>
/// <remarks>
/// <para>
/// An island is one independent sub-population with its own archive. Comparing these summaries is how you tell a
/// healthy island model from a degenerate one: islands that all report the same
/// <see cref="BestGenomeId"/> have converged on a single idea, and an island whose <see cref="EliteCount"/> stays
/// near zero while others fill up is being starved of proposals.
/// </para>
/// <para><b>For Beginners:</b> Splitting the search into islands is like running several small competitions instead
/// of one big one, letting each explore its own direction before the best members are exchanged. This tells you how
/// each competition went: how many distinct good answers it collected, which was its best, and how many rounds it
/// managed.</para>
/// </remarks>
public sealed class EvolutionIslandSummary
{
    /// <summary>Gets or sets the island index, counted from zero.</summary>
    public int Island { get; set; }

    /// <summary>Gets or sets how many variation proposals this island produced.</summary>
    public long Generation { get; set; }

    /// <summary>Gets or sets how many archive cells this island filled.</summary>
    public int EliteCount { get; set; }

    /// <summary>Gets or sets how many cells the archive grid has in total.</summary>
    public long TotalCells { get; set; }

    /// <summary>Gets or sets the fraction of the grid this island filled, from 0 to 1.</summary>
    public double Coverage { get; set; }

    /// <summary>Gets or sets the identifier of this island's best entry, or <c>null</c> when it archived nothing.</summary>
    public string? BestGenomeId { get; set; }

    /// <summary>Gets or sets this island's best quality, or <c>null</c> when it archived nothing scored.</summary>
    public double? BestQuality { get; set; }

    /// <summary>Gets or sets the mean quality across this island's entries, or <c>null</c> when there are none.</summary>
    public double? MeanQuality { get; set; }

    /// <summary>Gets or sets how many past evaluations this island retained beyond its elites.</summary>
    public int HistoryCount { get; set; }
}
