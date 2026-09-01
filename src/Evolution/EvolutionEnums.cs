namespace AiDotNet.Evolution;

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

/// <summary>Specifies whether larger or smaller scalar quality values are preferred.</summary>
public enum EvolutionOptimizationDirection
{
    /// <summary>Larger quality values are better.</summary>
    Maximize = 0,
    /// <summary>Smaller quality values are better.</summary>
    Minimize = 1
}

/// <summary>Controls how descriptor values outside configured bounds are binned.</summary>
public enum EvolutionOutOfRangePolicy
{
    /// <summary>Reject the candidate when a descriptor is outside its configured range.</summary>
    Reject = 0,
    /// <summary>Clamp below-range and above-range values into the first and last bins.</summary>
    Clamp = 1,
    /// <summary>Reserve explicit bins below and above the configured range.</summary>
    OverflowBins = 2
}

/// <summary>Describes how completed worker results are committed to evolution state.</summary>
public enum EvolutionExecutionMode
{
    /// <summary>Commit in evaluation-ID order so worker timing cannot change the result.</summary>
    Deterministic = 0,
    /// <summary>Commit as workers finish; this can improve responsiveness but is schedule-dependent.</summary>
    Opportunistic = 1
}

/// <summary>Controls whether an individual candidate failure stops the run.</summary>
public enum EvolutionFailurePolicy
{
    /// <summary>Record the failure and continue evaluating unrelated candidates.</summary>
    Continue = 0,
    /// <summary>Stop the run after the first recoverable candidate failure.</summary>
    FailFast = 1
}

/// <summary>Reports the result of attempting to add a completed evaluation to an archive.</summary>
public enum EvolutionArchiveInsertionResult
{
    /// <summary>The candidate filled an empty cell.</summary>
    Inserted = 0,
    /// <summary>The candidate replaced the incumbent in its cell.</summary>
    Replaced = 1,
    /// <summary>The candidate was not better than the incumbent.</summary>
    NotImproved = 2,
    /// <summary>The candidate or one of its descriptors was invalid for this archive.</summary>
    Rejected = 3,
    /// <summary>The candidate was inserted and a deterministic capacity eviction occurred.</summary>
    InsertedWithEviction = 4
}

/// <summary>Identifies whether an evaluation came from the task or a deterministic cache.</summary>
public enum EvolutionCacheStatus
{
    /// <summary>No cache lookup was performed.</summary>
    NotChecked = 0,
    /// <summary>The cache did not contain a reusable evaluation.</summary>
    Miss = 1,
    /// <summary>A prior completed evaluation was reused.</summary>
    Hit = 2
}

/// <summary>Explains why an evolution run returned control to its caller.</summary>
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
    GenerationLimitReached = 6
}

/// <summary>Classifies observer notifications emitted by the engine.</summary>
public enum EvolutionEventKind
{
    /// <summary>A candidate was proposed and assigned an identity.</summary>
    Proposed = 0,
    /// <summary>An evaluation reached a terminal status.</summary>
    Evaluated = 1,
    /// <summary>An archive accepted or replaced an elite.</summary>
    ArchiveChanged = 2,
    /// <summary>Migration copied elites between islands.</summary>
    Migrated = 3,
    /// <summary>A checkpoint was durably published.</summary>
    Checkpointed = 4,
    /// <summary>The run stopped.</summary>
    Stopped = 5
}
