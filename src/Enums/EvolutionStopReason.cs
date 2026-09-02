namespace AiDotNet.Enums;

/// <summary>Explains why an evolution run returned control to its caller.</summary>
/// <remarks>
/// <para>
/// <c>EvolutionEngine&lt;TGenome&gt;.RunAsync</c> stops at the first budget, limit, or policy that applies and reports
/// the reason through the run result and the final <c>Stopped</c> event. Budgets are checked in a fixed order before
/// each proposal (evaluation attempts, then proposals, then wall-clock time), so the reported reason is deterministic
/// for a given seed and configuration. <see cref="Canceled"/> appears only in the <c>Stopped</c> event: when the
/// caller's token is canceled the engine saves a final checkpoint and then rethrows
/// <c>OperationCanceledException</c> instead of returning a result. <see cref="TargetReached"/> and
/// <see cref="EarlyStopped"/> are checked after each committed batch, in that order, and a final checkpoint is always
/// written before the run returns - unlike OpenEvolve, whose controller writes the final checkpoint only when the last
/// iteration index is a multiple of the checkpoint interval (controller.py:508-513).
/// </para>
/// <para><b>For Beginners:</b> When an evolution run finishes, this value is the one-word answer to "why did it
/// stop?". Most of the time you will see a budget reason such as <see cref="EvaluationBudgetReached"/>, which simply
/// means the run used up the evaluations you allowed and is a normal ending. Reasons like <see cref="NoCandidates"/> or
/// <see cref="CandidateFailure"/> point at something worth investigating: either the archives held nothing that the
/// selection policy could use as a parent, or a fail-fast policy stopped the run on the first evaluator failure. Check
/// this value before reading the archives so you know whether the results reflect a completed search or an interrupted
/// one.</para>
/// </remarks>
public enum EvolutionStopReason
{
    /// <summary>The configured evaluation-attempt budget was consumed.</summary>
    EvaluationBudgetReached = 0,
    /// <summary>The configured proposal budget was consumed.</summary>
    ProposalBudgetReached = 1,
    /// <summary>No valid parent or initial candidate remained.</summary>
    NoCandidates = 2,
    /// <summary>The caller canceled the run.</summary>
    Canceled = 3,
    /// <summary>The configured wall-clock time limit elapsed.</summary>
    TimeLimitReached = 4,
    /// <summary>A fail-fast policy stopped the run.</summary>
    CandidateFailure = 5,
    /// <summary>The run completed its configured generation limit.</summary>
    GenerationLimitReached = 6,
    /// <summary>The configured target quality was reached in the archive's optimization direction.</summary>
    TargetReached = 7,
    /// <summary>The configured early-stopping metric plateaued for the configured patience.</summary>
    EarlyStopped = 8
}
