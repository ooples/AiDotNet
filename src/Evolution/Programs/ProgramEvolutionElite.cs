using System.Collections.ObjectModel;
using System.Globalization;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.Validation;

namespace AiDotNet.Evolution.Programs;

/// <summary>One archived program from a finished run, with its score, its cell, and a bounded copy of its source.</summary>
/// <remarks>
/// <para>
/// A MAP-Elites archive keeps the best program found for every distinct behaviour cell rather than a single winner,
/// so the useful output of a run is a list of these, not one program. <see cref="Cell"/> holds the bin indices that
/// placed this program in the grid and <see cref="Descriptors"/> holds the raw coordinate values behind them, which
/// together explain why two elites with similar scores were both kept.
/// </para>
/// <para>
/// <see cref="Source"/> is a bounded copy: it is the program text truncated to the limit the result was built with,
/// and <see cref="IsSourceTruncated"/> says whether anything was cut. That bound exists because a run's summary is
/// routinely logged or serialized, and program text is model-generated content of unbounded size. When the whole
/// program is needed, read it from the archive entry the result was built from rather than from here.
/// </para>
/// <para><b>For Beginners:</b> When the search finishes it does not hand back one answer; it hands back a map of the
/// best answer it found in each style or size of solution. This type is one square of that map: the program, how
/// well it scored, and where it sat on the map. Reading several of them side by side is usually more informative
/// than reading the single best one, because it shows you the trade-offs the search discovered.</para>
/// </remarks>
public sealed class ProgramEvolutionElite
{
    private static readonly ReadOnlyDictionary<string, double> NoDescriptors =
        new(new Dictionary<string, double>(StringComparer.Ordinal));

    private readonly ReadOnlyCollection<int> _cell;
    private readonly ReadOnlyDictionary<string, double> _descriptors;

    /// <summary>Initializes an elite summary.</summary>
    /// <param name="genomeId">The program's canonical identity.</param>
    /// <param name="source">The bounded program text.</param>
    /// <param name="isSourceTruncated">Whether <paramref name="source"/> was cut to fit the bound.</param>
    /// <param name="language">The program's language.</param>
    /// <param name="quality">Its fitness score, or <c>null</c> when it was never scored.</param>
    /// <param name="descriptors">Its raw archive coordinates, or <c>null</c> for none.</param>
    /// <param name="cell">The bin indices that placed it in the grid, or <c>null</c> for none.</param>
    /// <param name="island">The zero-based island the program was archived on.</param>
    /// <param name="evaluationId">The evaluation that produced the score.</param>
    /// <exception cref="ArgumentNullException"><paramref name="source"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="genomeId"/> is empty, or a descriptor value is not finite.</exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="quality"/> is not finite, <paramref name="island"/> or <paramref name="evaluationId"/> is
    /// negative, or a cell index is negative.
    /// </exception>
    public ProgramEvolutionElite(
        string genomeId,
        string source,
        bool isSourceTruncated,
        ProgramLanguage language,
        double? quality,
        IReadOnlyDictionary<string, double>? descriptors,
        IReadOnlyList<int>? cell,
        int island,
        long evaluationId)
    {
        Guard.NotNullOrWhiteSpace(genomeId);
        Guard.NotNull(source);
        if (island < 0) throw new ArgumentOutOfRangeException(nameof(island), island, "Value cannot be negative.");
        if (evaluationId < 0) throw new ArgumentOutOfRangeException(nameof(evaluationId), evaluationId, "Value cannot be negative.");
        if (quality.HasValue && (double.IsNaN(quality.Value) || double.IsInfinity(quality.Value)))
        {
            throw new ArgumentOutOfRangeException(nameof(quality), quality.Value, "Value must be a finite number.");
        }

        var cellCopy = new List<int>();
        if (cell is not null)
        {
            foreach (int bin in cell)
            {
                if (bin < 0) throw new ArgumentOutOfRangeException(nameof(cell), bin, "A cell index cannot be negative.");
                cellCopy.Add(bin);
            }
        }

        GenomeId = genomeId.Trim();
        Source = source;
        IsSourceTruncated = isSourceTruncated;
        Language = language;
        Quality = quality;
        Island = island;
        EvaluationId = evaluationId;
        _cell = new ReadOnlyCollection<int>(cellCopy);
        _descriptors = descriptors is null ? NoDescriptors : CopyDescriptors(descriptors);
    }

    /// <summary>Gets the program's canonical identity, the hash of its normalized source.</summary>
    public string GenomeId { get; }

    /// <summary>Gets the program text, truncated to the bound the result was built with.</summary>
    public string Source { get; }

    /// <summary>Gets whether <see cref="Source"/> was cut to fit that bound.</summary>
    public bool IsSourceTruncated { get; }

    /// <summary>Gets the program's language.</summary>
    public ProgramLanguage Language { get; }

    /// <summary>Gets the fitness score, or <c>null</c> when the program was never scored.</summary>
    public double? Quality { get; }

    /// <summary>Gets the raw archive coordinates behind <see cref="Cell"/>; empty when none were recorded.</summary>
    public IReadOnlyDictionary<string, double> Descriptors => _descriptors;

    /// <summary>Gets the bin indices that placed this program in the grid; empty when none were recorded.</summary>
    public IReadOnlyList<int> Cell => _cell;

    /// <summary>Gets the zero-based island the program was archived on.</summary>
    public int Island { get; }

    /// <summary>Gets the identifier of the evaluation that produced the score.</summary>
    public long EvaluationId { get; }

    /// <summary>Returns a description that never echoes the program text.</summary>
    /// <returns>The identity prefix, the score, and the cell.</returns>
    public override string ToString() =>
        "ProgramEvolutionElite(" + (GenomeId.Length > 12 ? GenomeId.Substring(0, 12) : GenomeId) +
        ", quality=" + (Quality.HasValue ? Quality.Value.ToString("R", CultureInfo.InvariantCulture) : "none") +
        ", cell=[" + string.Join(",", _cell.Select(bin => bin.ToString(CultureInfo.InvariantCulture))) + "])";

    private static ReadOnlyDictionary<string, double> CopyDescriptors(IReadOnlyDictionary<string, double> source)
    {
        var copy = new Dictionary<string, double>(StringComparer.Ordinal);
        foreach (KeyValuePair<string, double> pair in source)
        {
            if (pair.Key is null) throw new ArgumentException("A descriptor name cannot be null.", nameof(source));
            if (double.IsNaN(pair.Value) || double.IsInfinity(pair.Value))
            {
                throw new ArgumentException($"Descriptor '{pair.Key}' must be a finite number.", nameof(source));
            }

            copy[pair.Key] = pair.Value;
        }

        return new ReadOnlyDictionary<string, double>(copy);
    }
}
