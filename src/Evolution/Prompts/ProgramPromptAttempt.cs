using AiDotNet.Validation;

namespace AiDotNet.Evolution.Prompts;

/// <summary>One earlier attempt summarized for the prompt: what it changed, what it measured, how it compared.</summary>
/// <remarks>
/// <para>
/// The previous-attempts section is what stops a search from proposing the same failed idea repeatedly. Each
/// entry pairs the change that was made with the numbers it produced and the numbers its own parent produced, so
/// the prompt can state the outcome — everything improved, everything regressed, or a mix — rather than leaving
/// the model to diff two lists of floating-point values in its head.
/// </para>
/// <para>
/// Carrying <see cref="ParentMetrics"/> explicitly is what makes that verdict correct. The comparison is against
/// the attempt's own parent, not against the current program, which are different baselines whenever the search
/// has moved on since the attempt was made.
/// </para>
/// <para><b>For Beginners:</b> This is one line of history: "on attempt 7 we tried caching the results, the score
/// went from 0.61 to 0.58, so it was a step backwards". Feeding a few of these into the next prompt is what keeps
/// the AI from suggesting the same thing again. You supply the numbers from the attempt and the numbers from
/// whatever it was derived from, and the library works out the verdict.</para>
/// </remarks>
public sealed class ProgramPromptAttempt
{
    /// <summary>The longest change description accepted, in characters.</summary>
    public const int MaxChangesLength = 4_096;

    private static readonly Dictionary<string, double> NoMetrics = new(StringComparer.Ordinal);

    /// <summary>Initializes an attempt summary.</summary>
    /// <param name="attemptNumber">The one-based position of this attempt in the run.</param>
    /// <param name="changesDescription">What the attempt changed, or <c>null</c> when it was not recorded.</param>
    /// <param name="metrics">The values the attempt measured, or <c>null</c> for none.</param>
    /// <param name="parentMetrics">The values its parent measured, or <c>null</c> when unknown.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="attemptNumber"/> is not positive.</exception>
    /// <exception cref="ArgumentException"><paramref name="changesDescription"/> exceeds <see cref="MaxChangesLength"/> or a metric name is <c>null</c>.</exception>
    public ProgramPromptAttempt(
        int attemptNumber,
        string? changesDescription = null,
        IReadOnlyDictionary<string, double>? metrics = null,
        IReadOnlyDictionary<string, double>? parentMetrics = null)
    {
        Guard.Positive(attemptNumber);
        if (changesDescription is { } description && description.Length > MaxChangesLength)
        {
            throw new ArgumentException(
                $"An attempt's change description cannot exceed {MaxChangesLength} characters.", nameof(changesDescription));
        }

        AttemptNumber = attemptNumber;
        ChangesDescription = changesDescription;
        Metrics = metrics is null ? NoMetrics : CopyOf(metrics, nameof(metrics));
        ParentMetrics = parentMetrics is null ? NoMetrics : CopyOf(parentMetrics, nameof(parentMetrics));
    }

    /// <summary>Gets the one-based position of this attempt in the run.</summary>
    public int AttemptNumber { get; }

    /// <summary>Gets what the attempt changed, or <c>null</c> when it was not recorded.</summary>
    public string? ChangesDescription { get; }

    /// <summary>Gets the values the attempt measured; empty when none were supplied.</summary>
    public IReadOnlyDictionary<string, double> Metrics { get; }

    /// <summary>Gets the values the attempt's parent measured; empty when unknown.</summary>
    public IReadOnlyDictionary<string, double> ParentMetrics { get; }

    /// <summary>Returns a description that never echoes the change text.</summary>
    /// <returns>The attempt number and metric count.</returns>
    public override string ToString() =>
        $"ProgramPromptAttempt(#{AttemptNumber}, metrics={Metrics.Count})";

    private static Dictionary<string, double> CopyOf(IReadOnlyDictionary<string, double> source, string parameterName)
    {
        var copy = new Dictionary<string, double>(StringComparer.Ordinal);
        foreach (KeyValuePair<string, double> pair in source)
        {
            if (pair.Key is null) throw new ArgumentException("A metric name cannot be null.", parameterName);
            copy[pair.Key] = pair.Value;
        }

        return copy;
    }
}
