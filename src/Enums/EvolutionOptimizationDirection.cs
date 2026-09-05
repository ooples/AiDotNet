namespace AiDotNet.Enums;

/// <summary>Specifies whether larger or smaller scalar quality values are preferred.</summary>
/// <remarks>
/// <para>
/// Every evaluation result carries a direction alongside its scalar quality, and every archive is configured with the
/// direction it expects. The built-in MAP-Elites archive uses its direction to rank two candidates that land in the same
/// cell and keeps the better one; it rejects, rather than mis-ranks, an evaluation whose direction differs from its own,
/// and it folds the direction into its definition hash so checkpoints written under one direction cannot be resumed
/// under the other.
/// </para>
/// <para><b>For Beginners:</b> Some scores are "higher is better" (accuracy, F1, reward) and some are "lower is
/// better" (loss, error, latency, cost). This enum tells the evolution archive which way is up so it keeps the right
/// elite in each cell. For example, if your task scores candidates by validation loss, return
/// <see cref="Minimize"/>; if it scores them by test accuracy, return <see cref="Maximize"/>. Use the same value in the
/// task's results and in the archive's configuration, because the archive discards results whose direction does not
/// match its own.</para>
/// </remarks>
public enum EvolutionOptimizationDirection
{
    /// <summary>Larger quality values are better.</summary>
    Maximize = 0,
    /// <summary>Smaller quality values are better.</summary>
    Minimize = 1
}
