namespace AiDotNet.Enums;

/// <summary>Describes the terminal outcome of one candidate evaluation.</summary>
public enum EvolutionEvaluationStatus
{
    /// <summary>The evaluator produced a usable quality and descriptor set.</summary>
    Completed = 0,
    /// <summary>The candidate violated a task constraint before an expensive evaluation completed.</summary>
    Rejected = 1,
    /// <summary>An equivalent canonical genome was already observed.</summary>
    Duplicate = 2,
    /// <summary>A configured pre-screen deliberately omitted the expensive evaluation.</summary>
    Skipped = 3,
    /// <summary>The candidate failed with a recoverable error.</summary>
    Failed = 4,
    /// <summary>The candidate was canceled.</summary>
    Canceled = 5,
    /// <summary>The candidate exceeded its evaluation timeout.</summary>
    TimedOut = 6
}
