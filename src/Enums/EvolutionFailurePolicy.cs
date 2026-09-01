namespace AiDotNet.Enums;

/// <summary>Controls whether an individual candidate failure stops the run.</summary>
public enum EvolutionFailurePolicy
{
    /// <summary>Record the failure and continue evaluating unrelated candidates.</summary>
    Continue = 0,
    /// <summary>Stop the run after the first recoverable candidate failure.</summary>
    FailFast = 1
}
