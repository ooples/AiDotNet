namespace AiDotNet.Enums;

/// <summary>Controls whether an individual candidate failure stops the run.</summary>
/// <remarks>
/// <para>
/// The policy is read from <c>EvolutionEngineOptions.FailurePolicy</c> when a candidate's terminal result is
/// committed, that is, after any configured retries have been used. Only
/// <see cref="EvolutionEvaluationStatus.Failed"/> and <see cref="EvolutionEvaluationStatus.TimedOut"/>
/// trigger fail-fast behaviour; <see cref="EvolutionEvaluationStatus.Canceled"/> never does. Failure
/// diagnostics are retained in the run state under either setting, and the chosen value participates in the
/// checkpoint compatibility hash, so changing it starts a new checkpoint lineage instead of resuming an old one.
/// </para>
/// <para><b>For Beginners:</b> When an evolutionary search evaluates hundreds of candidate solutions, a few
/// of them will occasionally crash or time out, for example because a proposed configuration is numerically
/// unstable. This setting decides what happens next. <see cref="Continue"/> treats such a failure like one
/// bad exam result: it is recorded against that candidate and the search moves on, which is what you want
/// for long, unattended runs. <see cref="FailFast"/> stops the whole run at the first failure, which is
/// useful while you are still debugging your evaluator and would rather see a problem immediately than find
/// it buried in a log hours later.</para>
/// </remarks>
public enum EvolutionFailurePolicy
{
    /// <summary>Record the failure and continue evaluating unrelated candidates.</summary>
    Continue = 0,
    /// <summary>Stop the run after the first recoverable candidate failure.</summary>
    FailFast = 1
}
