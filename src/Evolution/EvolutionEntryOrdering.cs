using AiDotNet.Enums;

namespace AiDotNet.Evolution;

/// <summary>
/// Shared deterministic best-first ordering for archive entries, used by the archive, the global elite index, and
/// the bounded island history so all three agree on which entry is "better".
/// </summary>
internal static class EvolutionEntryOrdering
{
    /// <summary>Creates a comparer that orders entries best first for the supplied direction.</summary>
    internal static IComparer<EvolutionArchiveEntry<TGenome>> BestFirst<TGenome>(EvolutionOptimizationDirection direction) =>
        new EntryComparer<TGenome>(direction);

    /// <summary>Compares two entries best first, breaking ties by genome identifier, cell, then evaluation identifier.</summary>
    internal static int Compare<TGenome>(EvolutionOptimizationDirection direction,
        EvolutionArchiveEntry<TGenome>? x, EvolutionArchiveEntry<TGenome>? y)
    {
        if (ReferenceEquals(x, y)) return 0;
        if (x is null) return 1;
        if (y is null) return -1;
        int quality = direction == EvolutionOptimizationDirection.Maximize
            ? Nullable.Compare(y.Evaluation.Quality, x.Evaluation.Quality)
            : Nullable.Compare(x.Evaluation.Quality, y.Evaluation.Quality);
        if (quality != 0) return quality;
        int genome = StringComparer.Ordinal.Compare(x.Evaluation.GenomeId, y.Evaluation.GenomeId);
        if (genome != 0) return genome;
        int cell = x.Cell.CompareTo(y.Cell);
        if (cell != 0) return cell;
        return x.Evaluation.EvaluationId.CompareTo(y.Evaluation.EvaluationId);
    }

    private sealed class EntryComparer<TGenome> : IComparer<EvolutionArchiveEntry<TGenome>>
    {
        private readonly EvolutionOptimizationDirection _direction;

        internal EntryComparer(EvolutionOptimizationDirection direction) => _direction = direction;

        public int Compare(EvolutionArchiveEntry<TGenome>? x, EvolutionArchiveEntry<TGenome>? y) =>
            EvolutionEntryOrdering.Compare(_direction, x, y);
    }
}
