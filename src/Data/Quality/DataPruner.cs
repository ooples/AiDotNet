namespace AiDotNet.Data.Quality;

/// <summary>
/// Prunes (removes) training samples based on difficulty/importance scores.
/// </summary>
/// <remarks>
/// <para>
/// Data pruning removes easy or redundant samples to reduce training set size
/// while preserving model quality. Supports multiple scoring strategies:
/// confidence-based, forgetting events, EL2N, and GraNd scores.
/// Requires per-sample training signals collected during a warmup phase.
/// </para>
/// </remarks>
public class DataPruner
{
    private readonly DataPrunerOptions _options;

    public DataPruner(DataPrunerOptions? options = null)
    {
        _options = options ?? new DataPrunerOptions();
        _options.Validate();
    }

    /// <summary>
    /// Identifies samples to prune based on confidence scores.
    /// Samples with consistently high confidence are considered easy and prunable.
    /// </summary>
    /// <param name="confidenceScores">Per-sample average confidence across epochs. Shape: [numSamples].</param>
    /// <returns>Set of indices to remove.</returns>
    public HashSet<int> PruneByConfidence(double[] confidenceScores)
    {
        if (confidenceScores == null) throw new ArgumentNullException(nameof(confidenceScores));
        int numToPrune = (int)(confidenceScores.Length * _options.PruneRatio);
        return PruneTopK(confidenceScores, numToPrune, ascending: false);
    }

    /// <summary>
    /// Identifies samples to prune based on forgetting event counts.
    /// Samples that are never forgotten during training are prunable.
    /// </summary>
    /// <param name="forgettingCounts">Number of times each sample was forgotten (learned then misclassified). Shape: [numSamples].</param>
    /// <returns>Set of indices to remove.</returns>
    public HashSet<int> PruneByForgetting(int[] forgettingCounts)
    {
        if (forgettingCounts == null) throw new ArgumentNullException(nameof(forgettingCounts));
        int numToPrune = (int)(forgettingCounts.Length * _options.PruneRatio);
        var scores = forgettingCounts.Select(c => (double)c).ToArray();
        return PruneTopK(scores, numToPrune, ascending: true);
    }

    /// <summary>
    /// Identifies samples to prune based on EL2N (Error L2 Norm) scores.
    /// Low EL2N samples are easy examples that contribute less to learning.
    /// </summary>
    /// <param name="el2nScores">Per-sample EL2N scores averaged over early epochs. Shape: [numSamples].</param>
    /// <returns>Set of indices to remove.</returns>
    public HashSet<int> PruneByEL2N(double[] el2nScores)
    {
        if (el2nScores == null) throw new ArgumentNullException(nameof(el2nScores));
        int numToPrune = (int)(el2nScores.Length * _options.PruneRatio);
        return PruneTopK(el2nScores, numToPrune, ascending: true);
    }

    /// <summary>
    /// Identifies samples to prune based on GraNd (Gradient Norm) scores.
    /// Low gradient norm samples have small impact on model updates.
    /// </summary>
    /// <param name="grandScores">Per-sample gradient norm scores. Shape: [numSamples].</param>
    /// <returns>Set of indices to remove.</returns>
    public HashSet<int> PruneByGraNd(double[] grandScores)
    {
        if (grandScores == null) throw new ArgumentNullException(nameof(grandScores));
        int numToPrune = (int)(grandScores.Length * _options.PruneRatio);
        return PruneTopK(grandScores, numToPrune, ascending: true);
    }

    /// <summary>
    /// Identifies samples to prune using the configured strategy.
    /// </summary>
    /// <param name="scores">Per-sample scores appropriate for the configured strategy.</param>
    /// <returns>Set of indices to remove.</returns>
    public HashSet<int> Prune(double[] scores)
    {
        if (scores == null) throw new ArgumentNullException(nameof(scores));
        return _options.Strategy switch
        {
            PruneStrategy.HighConfidence => PruneByConfidence(scores),
            PruneStrategy.NeverForgotten => PruneByForgetting(scores.Select(s => (int)s).ToArray()),
            PruneStrategy.LowEL2N => PruneByEL2N(scores),
            PruneStrategy.LowGraNd => PruneByGraNd(scores),
            _ => PruneByConfidence(scores)
        };
    }

    private static HashSet<int> PruneTopK(double[] scores, int k, bool ascending)
    {
        var indexed = scores
            .Select((score, idx) => (Score: score, Index: idx));

        var sorted = ascending
            ? indexed.OrderBy(x => x.Score)
            : indexed.OrderByDescending(x => x.Score);

        return new HashSet<int>(sorted.Take(k).Select(x => x.Index));
    }
}
