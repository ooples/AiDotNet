namespace AiDotNet.Enums;

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
