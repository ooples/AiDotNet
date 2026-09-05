using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>An immutable per-island snapshot of progress, occupancy, and quality.</summary>
/// <remarks>
/// <para>
/// One instance describes one island at the moment a run ended: how many variation proposals it has produced
/// (<see cref="Generation"/>), how many cells of its grid are occupied (<see cref="EliteCount"/> out of
/// <see cref="TotalCells"/>, expressed as <see cref="Coverage"/>), which elite currently leads it
/// (<see cref="BestGenomeId"/> and <see cref="BestQuality"/>), the mean quality of its elites
/// (<see cref="MeanQuality"/>), and how many entries its optional bounded history retains
/// (<see cref="HistoryCount"/>). Statistics are computed over every elite rather than over a five-program sample the
/// way OpenEvolve's island status does, so the numbers are exact.
/// </para>
/// <para><b>For Beginners:</b> When a search that uses several islands finishes, you usually want to know how each
/// one did rather than only the single overall winner. This is that per-island report card. Coverage tells you how
/// much of the behaviour map that island managed to fill, from 0 for empty to 1 for completely full; the generation
/// count tells you how much work went into it; and the best and mean quality tell you how strong its solutions are.
/// A low coverage with a high best quality usually means the island converged early, which is a hint to raise the
/// exploration ratio or add more islands.</para>
/// </remarks>
public sealed class EvolutionIslandStatus
{
    /// <summary>Initializes an island status snapshot.</summary>
    /// <param name="island">The zero-based island index.</param>
    /// <param name="generation">The number of variation proposals assigned to the island.</param>
    /// <param name="eliteCount">The number of occupied cells.</param>
    /// <param name="totalCells">The number of physical cells in the island's descriptor grid.</param>
    /// <param name="bestGenomeId">The canonical identifier of the leading elite, or <c>null</c> when empty.</param>
    /// <param name="bestQuality">The quality of the leading elite, or <c>null</c> when empty.</param>
    /// <param name="meanQuality">The mean quality across occupied cells, or <c>null</c> when empty.</param>
    /// <param name="historyCount">The number of entries retained by the island's bounded history.</param>
    /// <exception cref="ArgumentOutOfRangeException">A count is negative or <paramref name="totalCells"/> is not positive.</exception>
    public EvolutionIslandStatus(int island, long generation, int eliteCount, long totalCells,
        string? bestGenomeId, double? bestQuality, double? meanQuality, int historyCount)
    {
        if (island < 0) throw new ArgumentOutOfRangeException(nameof(island));
        if (generation < 0) throw new ArgumentOutOfRangeException(nameof(generation));
        Guard.NonNegative(eliteCount);
        if (totalCells <= 0) throw new ArgumentOutOfRangeException(nameof(totalCells));
        Guard.NonNegative(historyCount);
        Island = island;
        Generation = generation;
        EliteCount = eliteCount;
        TotalCells = totalCells;
        BestGenomeId = bestGenomeId;
        BestQuality = bestQuality;
        MeanQuality = meanQuality;
        HistoryCount = historyCount;
    }

    /// <summary>Gets the zero-based island index.</summary>
    public int Island { get; }

    /// <summary>Gets the number of variation proposals assigned to this island.</summary>
    public long Generation { get; }

    /// <summary>Gets the number of occupied cells.</summary>
    public int EliteCount { get; }

    /// <summary>Gets the number of physical cells in the island's descriptor grid.</summary>
    public long TotalCells { get; }

    /// <summary>Gets the occupied fraction of the descriptor grid, from zero to one.</summary>
    public double Coverage => EliteCount / (double)TotalCells;

    /// <summary>Gets the canonical identifier of the leading elite, or <c>null</c> when the island is empty.</summary>
    public string? BestGenomeId { get; }

    /// <summary>Gets the quality of the leading elite, or <c>null</c> when the island is empty.</summary>
    public double? BestQuality { get; }

    /// <summary>Gets the mean quality across occupied cells, or <c>null</c> when the island is empty.</summary>
    public double? MeanQuality { get; }

    /// <summary>Gets the number of entries retained by the island's bounded history.</summary>
    public int HistoryCount { get; }
}
