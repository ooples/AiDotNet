namespace AiDotNet.Enums;

/// <summary>Describes the terminal outcome of one candidate evaluation.</summary>
/// <remarks>
/// <para>
/// Only <see cref="Completed"/> carries a usable quality and descriptor set; archives reject every other status.
/// <see cref="Rejected"/>, <see cref="Duplicate"/>, and <see cref="Skipped"/> are inexpensive early exits decided
/// before the expensive evaluation runs, whereas <see cref="Failed"/>, <see cref="Canceled"/>, and
/// <see cref="TimedOut"/> describe an evaluation that started and did not finish normally. The status is stored on
/// <c>EvolutionEvaluation.Status</c> and is what the engine's retry, budget, and cache policies branch on: a
/// <see cref="Failed"/> or <see cref="TimedOut"/> candidate may be retried, while the others are final.
/// </para>
/// <para><b>For Beginners:</b> Every candidate the engine proposes ends up with exactly one of these outcomes,
/// much like the result of a job application. <see cref="Completed"/> means the candidate was fully evaluated and
/// received a score. <see cref="Rejected"/> means it broke a task rule (for example, an invalid hyperparameter
/// combination) before any expensive work happened, <see cref="Duplicate"/> means an identical candidate was
/// already seen, and <see cref="Skipped"/> means a cheap pre-screen decided it was not worth evaluating. The
/// remaining values tell you that the evaluation itself went wrong: it threw a recoverable error, was canceled, or
/// ran out of time. When you read run statistics, remember that only <see cref="Completed"/> evaluations can
/// appear in the archive.</para>
/// </remarks>
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
