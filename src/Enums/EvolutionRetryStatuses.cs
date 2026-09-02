namespace AiDotNet.Enums;

/// <summary>Selects which failure-like terminal statuses are eligible for another evaluation attempt.</summary>
/// <remarks>
/// <para>
/// The engine retries a candidate only when its status is in this set and the candidate is still within both
/// <c>EvolutionEngineOptions.MaxRetries</c> and the run's evaluation-attempt budget. The default
/// <see cref="All"/> preserves the engine's historical behaviour of retrying every failure-like outcome. Narrowing the
/// set is how a run stops paying for retries that cannot succeed: OpenEvolve hard-codes the equivalent choice by
/// returning immediately on a timeout and retrying only exceptions (evaluator.py:252-285), which this flag makes
/// explicit and configurable rather than implicit.
/// </para>
/// <para><b>For Beginners:</b> When scoring a candidate fails, the engine can try again. Sometimes that is worth it -
/// a flaky test or a transient file lock will often succeed on the second attempt. Sometimes it is pure waste - a
/// candidate that ran out of time will almost certainly run out of time again, and each retry costs you one of the
/// evaluations in your budget. This setting is where you say which kinds of failure deserve another go. Combine the
/// values with <c>|</c>, for example <c>Failed | Canceled</c> to retry errors and cancellations but never timeouts.</para>
/// </remarks>
[Flags]
public enum EvolutionRetryStatuses
{
    /// <summary>No status is retried; every failure-like result is terminal on its first attempt.</summary>
    None = 0,
    /// <summary>Retry <see cref="EvolutionEvaluationStatus.Failed"/> results.</summary>
    Failed = 1,
    /// <summary>Retry <see cref="EvolutionEvaluationStatus.TimedOut"/> results.</summary>
    TimedOut = 2,
    /// <summary>Retry <see cref="EvolutionEvaluationStatus.Canceled"/> results.</summary>
    Canceled = 4,
    /// <summary>Retry every failure-like status, which is the engine's default behaviour.</summary>
    All = Failed | TimedOut | Canceled
}
